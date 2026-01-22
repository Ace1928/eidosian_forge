import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
class _FusedSequenceParallel:
    """Set up a communication ring and perform fused ops on it

    Stores the persistent state needed to support a ring of connections between
    processes, and the logic that can do fused comms + matmuls on it.

    We want to achieve overlap between:
    - a computation which reads from the data we received from a remote GPU
    - and the communication where we send some data to another GPU
    And in order to do that we need some staging buffers and a way to
    synchronize access to them across processes.

    To perform the communication over NVLink we make the processes exchange
    their staging buffers using IPC (Inter-Process Communication) handles, which
    "mounts"/"mmaps" an allocation on one GPU into the virtual address space of
    another GPU: the memory remains backed by the original GPU but the other GPU
    can access it as if it were local. We exchange these IPC handles using
    multiprocessing Connections (and the "reductions" provided by PyTorch),
    which we establish over UNIX domain sockets, whose addresses we exchange by
    using a ProcessGroup.

    To synchronize accesses we use a set of counters/sequence numbers that are
    also allocated in memory shared over IPC handles. Processes signal that they
    completed an operation by launching a kernel that increases that value, and
    they wait for anoher process to complete an operation by launching a kernel
    that busy-waits for that value to increase. Currently we implement these
    kernels manually, but on recent CUDA drivers (515.43.04+, corresponding to
    CUDA 11.7) we could use standard stream memory operations (see
    https://docs.nvidia.com/cuda/archive/11.7.0/cuda-driver-api/group__CUDA__MEMOP.html).

    We prefer to use these kernels (or the stream memory ops) over IPC events
    because IPC events require signaling between processes at launch time to
    ensure that the wait on one process occurs after the record on another
    process. This signaling means that _launching_ our fused operation becomes a
    synchronization barrier, which can increase the launch overhead. It would
    also behave differently from NCCL, where launching is async and all the
    synchronization happens on device in the kernels. A previous version of this
    code which uses IPC events can be found here:
    https://github.com/fairinternal/xformers/pull/504.

    """

    def __init__(self, device: torch.device, dtype: torch.dtype, group: dist.ProcessGroup, num_stripes: int):
        self.my_device = device
        self.dtype = dtype
        self.my_rank = group.rank()
        self.world_size = group.size()
        self.num_stripes = num_stripes
        self.my_device_capability = torch.cuda.get_device_capability(self.my_device)
        self.p2p_comms = init_ipc(group, self.my_device)
        self.next_stripe = 0
        self.next_seq_nums = [1] * self.num_stripes
        self.staging = torch.empty((0,), device=self.my_device)
        self.buddys_staging = [torch.empty((0,), device=self.my_device)] * self.world_size
        self.num_writes_into_my_staging = torch.zeros((self.world_size, self.num_stripes), dtype=torch.int, device=self.my_device)
        self.num_reads_from_buddys_staging = torch.zeros((self.world_size, self.num_stripes), dtype=torch.int, device=self.my_device)
        for rank, conn in enumerate(self.p2p_comms):
            if conn is not None:
                conn.send(self.num_writes_into_my_staging[rank])
                conn.send(self.num_reads_from_buddys_staging[rank])
        self.num_writes_into_buddys_staging = [torch.empty((0,), device=self.my_device) if conn is None else conn.recv() for conn in self.p2p_comms]
        self.num_reads_from_my_staging = [torch.empty((0,), device=self.my_device) if conn is None else conn.recv() for conn in self.p2p_comms]
        self.second_stream = torch.cuda.Stream()
        self.memcpy_stream = torch.cuda.Stream(priority=-1)
        self.wait_stream = torch.cuda.Stream(priority=-1)
        self.write_stream = torch.cuda.Stream(priority=-1)
        self.next_stream_idx = 0

    def _ensure_staging_is_large_enough(self, num_elements: int, random_init: bool):
        if self.staging.numel() < self.world_size * num_elements:
            self.staging = torch.empty((self.num_stripes, self.world_size, num_elements), device=self.my_device, dtype=self.dtype)
            if random_init:
                self.staging.normal_()
            for rank, conn in enumerate(self.p2p_comms):
                if conn is not None:
                    conn.send(self.staging[:, rank])
            self.buddys_staging = [torch.empty((0,), device=self.my_device) if conn is None else conn.recv() for rank, conn in enumerate(self.p2p_comms)]

    def _should_use_triton(self, _triton: bool):
        if not int(os.getenv('XFORMERS_FUSED_SEQPAR_ENABLE_TRITON', '1')):
            return False
        if not TRITON_IS_AVAILABLE:
            return False
        if self.my_device_capability < (8, 0):
            return False
        if not _triton:
            return False
        return True

    def make_stream_factory(self, current_stream: torch.cuda.Stream) -> Callable[[], torch.cuda.Stream]:

        def result():
            stream = [current_stream, self.second_stream][self.next_stream_idx]
            self.next_stream_idx += 1
            self.next_stream_idx %= 2
            return stream
        return result

    def allgather_and_linear(self, scattered_inputs: List[torch.Tensor], my_matmul: Callable[[List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None], timeout_s: int, _wait: bool=True, _memcpy: bool=True, _triton: bool=True, _is_regular_matmul: bool=False, _extra_triton_args: Mapping[str, Any]={}):
        """Perform a fused all-gather followed by a linear layer"""
        assert all((si.device == self.my_device for si in scattered_inputs))
        assert all((si.dtype == self.dtype for si in scattered_inputs))
        scattered_input_numels = [si.numel() for si in scattered_inputs]
        total_scattered_input_numel = sum(scattered_input_numels)
        self._ensure_staging_is_large_enough(total_scattered_input_numel, random_init=_memcpy is False)
        stripe = self.next_stripe % self.num_stripes
        self.next_stripe += 1
        seq_num = self.next_seq_nums[stripe] % SEQ_NUM_WRAP_AROUND
        prev_seq_num = (seq_num - 1) % SEQ_NUM_WRAP_AROUND
        self.next_seq_nums[stripe] += 1
        stagings = [s.view((self.world_size,) + si.shape) for s, si in zip(self.staging[stripe, :, :total_scattered_input_numel].split(scattered_input_numels, dim=-1), scattered_inputs)]
        buddys_stagings = [[bs] * len(scattered_inputs) if bs.numel() == 0 else [s.view(si.shape) for s, si in zip(bs[stripe, :total_scattered_input_numel].split(scattered_input_numels, dim=-1), scattered_inputs)] for bs in self.buddys_staging]
        current_stream = torch.cuda.current_stream()
        self.memcpy_stream.wait_stream(current_stream)
        if _wait:
            WaitValues.OPERATOR([self.num_reads_from_buddys_staging[(self.my_rank + iter_) % self.world_size, stripe] for iter_ in range(1, self.world_size)], prev_seq_num, self.memcpy_stream, timeout_s)
        for iter_ in range(1, self.world_size):
            dst_rank = (self.my_rank + iter_) % self.world_size
            if _memcpy:
                with torch.cuda.stream(self.memcpy_stream):
                    for bs, si in zip(buddys_stagings[dst_rank], scattered_inputs):
                        bs.copy_(si)
            self.write_stream.wait_stream(self.memcpy_stream)
            if _wait:
                Memset32bAsync.OPERATOR(self.num_writes_into_buddys_staging[dst_rank][stripe], seq_num, self.write_stream)
        if _is_regular_matmul and self._should_use_triton(_triton):
            _launch_triton_matmul(a_my_shard=scattered_inputs[0].flatten(0, -2), a=stagings[0].flatten(0, -2), my_rank=self.my_rank, world_size=self.world_size, wait_counters=self.num_writes_into_my_staging, write_counters=None, direction=BACKWARDS_WITH_ME_FIRST, stripe=stripe, seq_num=seq_num, num_stripes=self.num_stripes, timeout_s=timeout_s, _wait=_wait, **_extra_triton_args)
        else:
            self.wait_stream.wait_stream(current_stream)
            self.second_stream.wait_stream(current_stream)
            stream_factory = self.make_stream_factory(current_stream)
            my_matmul(scattered_inputs, self.my_rank, stream_factory)
            for iter_ in range(1, self.world_size):
                src_rank = (self.my_rank - iter_) % self.world_size
                if _wait:
                    WaitValues.OPERATOR([self.num_writes_into_my_staging[src_rank, stripe]], seq_num, self.wait_stream, timeout_s)
                    current_stream.wait_stream(self.wait_stream)
                    self.second_stream.wait_stream(self.wait_stream)
                my_matmul([s[src_rank] for s in stagings], src_rank, stream_factory)
            current_stream.wait_stream(self.second_stream)
        self.write_stream.wait_stream(current_stream)
        if _wait:
            WriteValues.OPERATOR([self.num_reads_from_my_staging[(self.my_rank - iter_) % self.world_size][stripe] for iter_ in range(1, self.world_size)], seq_num, self.write_stream)

    def linear_and_reducescatter(self, my_matmul: Callable[[List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None], gathered_outputs: List[torch.Tensor], scattered_outputs: List[torch.Tensor], timeout_s: int, _wait: bool=True, _memcpy: bool=True, _triton: bool=True, _is_regular_matmul: bool=False, _extra_triton_args: Mapping[str, Any]={}):
        """Perform a fused linear layer followed by a reduce-scatter"""
        assert all((go.device == self.my_device for go in gathered_outputs))
        assert all((go.dtype == self.dtype for go in gathered_outputs))
        assert all((so.device == self.my_device for so in scattered_outputs))
        assert all((so.dtype == self.dtype for so in scattered_outputs))
        scattered_output_numels = [so.numel() for so in scattered_outputs]
        total_scattered_output_numel = sum(scattered_output_numels)
        self._ensure_staging_is_large_enough(total_scattered_output_numel, random_init=_memcpy is False)
        stripe = self.next_stripe % self.num_stripes
        self.next_stripe += 1
        seq_num = self.next_seq_nums[stripe] % SEQ_NUM_WRAP_AROUND
        prev_seq_num = (seq_num - 1) % SEQ_NUM_WRAP_AROUND
        self.next_seq_nums[stripe] += 1
        stagings = [s.view((self.world_size,) + so.shape) for s, so in zip(self.staging[stripe, :, :total_scattered_output_numel].split(scattered_output_numels, dim=-1), scattered_outputs)]
        buddys_stagings = [[bs] * len(scattered_outputs) if bs.numel() == 0 else [s.view(so.shape) for s, so in zip(bs[stripe, :total_scattered_output_numel].split(scattered_output_numels, dim=-1), scattered_outputs)] for bs in self.buddys_staging]
        current_stream = torch.cuda.current_stream()
        self.wait_stream.wait_stream(current_stream)
        if _wait:
            WaitValues.OPERATOR([self.num_reads_from_my_staging[(self.my_rank + iter_) % self.world_size][stripe] for iter_ in range(1, self.world_size)], prev_seq_num, current_stream, timeout_s)
        if _is_regular_matmul and self._should_use_triton(_triton):
            _launch_triton_matmul(cs=[s.flatten(0, -2) for s in stagings], cs_my_shard=[go[self.my_rank].flatten(0, -2) for go in gathered_outputs], my_rank=self.my_rank, world_size=self.world_size, wait_counters=None, write_counters=self.num_writes_into_my_staging, direction=FORWARDS_WITH_ME_LAST, stripe=stripe, seq_num=seq_num, num_stripes=self.num_stripes, timeout_s=timeout_s, _wait=_wait, **_extra_triton_args)
        else:
            self.second_stream.wait_stream(current_stream)
            stream_factory = self.make_stream_factory(current_stream)
            for iter_ in range(1, self.world_size):
                dst_rank = (self.my_rank + iter_) % self.world_size
                my_matmul([s[dst_rank] for s in stagings], dst_rank, stream_factory)
                if _wait:
                    self.write_stream.wait_stream(current_stream)
                    self.write_stream.wait_stream(self.second_stream)
                    WriteValues.OPERATOR([self.num_writes_into_my_staging[dst_rank, stripe]], seq_num, self.write_stream)
            my_matmul([o[self.my_rank] for o in gathered_outputs], self.my_rank, stream_factory)
            current_stream.wait_stream(self.second_stream)
        for iter_ in range(1, self.world_size):
            src_rank = (self.my_rank - iter_) % self.world_size
            if _wait:
                WaitValues.OPERATOR([self.num_writes_into_buddys_staging[src_rank][stripe]], seq_num, self.wait_stream, timeout_s)
            self.memcpy_stream.wait_stream(self.wait_stream)
            if _memcpy:
                with torch.cuda.stream(self.memcpy_stream):
                    for go, bs in zip(gathered_outputs, buddys_stagings[src_rank]):
                        go[src_rank].copy_(bs)
        current_stream.wait_stream(self.memcpy_stream)
        for go, so in zip(gathered_outputs, scattered_outputs):
            torch.sum(go, dim=0, out=so)
        self.write_stream.wait_stream(current_stream)
        if _wait:
            WriteValues.OPERATOR([self.num_reads_from_buddys_staging[(self.my_rank - iter_) % self.world_size, stripe] for iter_ in range(1, self.world_size)], seq_num, self.write_stream)