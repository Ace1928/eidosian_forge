from collections import deque
import enum
import time
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union, Set
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.block_manager import AllocStatus, BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.lora.request import LoRARequest
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
from vllm.prefix import PrefixPool
def _schedule(self) -> SchedulerOutputs:
    blocks_to_swap_in: Dict[int, int] = {}
    blocks_to_swap_out: Dict[int, int] = {}
    blocks_to_copy: Dict[int, List[int]] = {}
    now = time.monotonic()
    if not self.swapped:
        ignored_seq_groups: List[SequenceGroup] = []
        scheduled: List[SequenceGroup] = []
        num_curr_seqs = sum((seq_group.get_max_num_running_seqs() for seq_group in self.running))
        curr_loras = set((seq_group.lora_int_id for seq_group in self.running)) if self.lora_enabled else None
        seq_lens: List[int] = []
        leftover_waiting_sequences = deque()
        while self.waiting:
            seq_group = self.waiting[0]
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, 'Waiting sequence group should have only one prompt sequence.'
            num_prompt_tokens = waiting_seqs[0].get_len()
            if num_prompt_tokens > self.prompt_limit:
                logger.warning(f'Input prompt ({num_prompt_tokens} tokens) is too long and exceeds limit of {self.prompt_limit}')
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.popleft()
                continue
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(f'Input prompt ({num_prompt_tokens} tokens) is too long and exceeds the capacity of block_manager')
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.popleft()
                continue
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if lora_int_id > 0 and lora_int_id not in curr_loras and (len(curr_loras) >= self.lora_config.max_loras):
                    leftover_waiting_sequences.appendleft(seq_group)
                    self.waiting.popleft()
                    continue
            new_seq_lens = seq_lens + [num_prompt_tokens]
            num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
            if num_batched_tokens > self.scheduler_config.max_num_batched_tokens:
                break
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                break
            num_paddings = num_batched_tokens - sum(new_seq_lens)
            if num_paddings > self.scheduler_config.max_paddings:
                break
            seq_lens = new_seq_lens
            if lora_int_id > 0:
                curr_loras.add(lora_int_id)
            self.waiting.popleft()
            self._allocate(seq_group)
            self.running.append(seq_group)
            num_curr_seqs += num_new_seqs
            scheduled.append(seq_group)
        self.waiting.extendleft(leftover_waiting_sequences)
        if scheduled or ignored_seq_groups:
            scheduler_outputs = SchedulerOutputs(scheduled_seq_groups=scheduled, prompt_run=True, num_batched_tokens=len(seq_lens) * max(seq_lens) if seq_lens else 0, blocks_to_swap_in=blocks_to_swap_in, blocks_to_swap_out=blocks_to_swap_out, blocks_to_copy=blocks_to_copy, ignored_seq_groups=ignored_seq_groups)
            return scheduler_outputs
    self.running = self.policy.sort_by_priority(now, self.running)
    running: Deque[SequenceGroup] = deque()
    preempted: List[SequenceGroup] = []
    while self.running:
        seq_group = self.running.popleft()
        while not self.block_manager.can_append_slot(seq_group):
            if self.running:
                victim_seq_group = self.running.pop()
                self._preempt(victim_seq_group, blocks_to_swap_out)
                preempted.append(victim_seq_group)
            else:
                self._preempt(seq_group, blocks_to_swap_out)
                preempted.append(seq_group)
                break
        else:
            self._append_slot(seq_group, blocks_to_copy)
            running.append(seq_group)
    self.running = running
    self.swapped = self.policy.sort_by_priority(now, self.swapped)
    if not preempted:
        num_curr_seqs = sum((seq_group.get_max_num_running_seqs() for seq_group in self.running))
        curr_loras = set((seq_group.lora_int_id for seq_group in self.running)) if self.lora_enabled else None
        leftover_swapped = deque()
        while self.swapped:
            seq_group = self.swapped[0]
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if lora_int_id > 0 and lora_int_id not in curr_loras and (len(curr_loras) >= self.lora_config.max_loras):
                    leftover_swapped.appendleft(seq_group)
                    self.swapped.popleft()
                    continue
            if not self.block_manager.can_swap_in(seq_group):
                break
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                break
            if lora_int_id > 0:
                curr_loras.add(lora_int_id)
            self.swapped.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            num_curr_seqs += num_new_seqs
            self.running.append(seq_group)
        self.swapped.extendleft(leftover_swapped)
    num_batched_tokens = sum((seq_group.num_seqs(status=SequenceStatus.RUNNING) for seq_group in self.running))
    scheduler_outputs = SchedulerOutputs(scheduled_seq_groups=self.running, prompt_run=False, num_batched_tokens=num_batched_tokens, blocks_to_swap_in=blocks_to_swap_in, blocks_to_swap_out=blocks_to_swap_out, blocks_to_copy=blocks_to_copy, ignored_seq_groups=[])
    return scheduler_outputs