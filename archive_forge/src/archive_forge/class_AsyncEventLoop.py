from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple
import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup
from fairscale.nn.model_parallel import get_pipeline_parallel_ranks
from .checkpoint import Checkpointing
from .messages import Transport
from .microbatch import Batch
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors, Tensors
from .worker import Task
class AsyncEventLoop:

    def __init__(self, partitions: List[ModuleWrapper], group: ProcessGroup, transport: Transport, training: bool, checkpoint_stop: int):
        self.training = training
        self.checkpoint_stop = checkpoint_stop
        self.transport = transport
        self.group = group
        self.partitions: List[ModuleWrapper] = partitions

    def send_async_message(self, dst_rank: int, result: Batch, invocation: Invocation) -> Batch:
        """Send batch to dst_rank, and use AutogradWithoutActivations to delete
        the activations since we no longer need them"""
        assert invocation.dest
        src_rank = torch.distributed.get_rank()
        body = AsyncMessageBody(AsyncMessageType.Activations, result.index, invocation.this, invocation.dest, invocation.order + 1)
        self.transport.send_message(PipeMessage(src_rank, dst_rank, queue_name=EVENT_LOOP_QUEUE, args=body, tensors=tuple([*result])), sync=True)
        phony = AutogradWithoutActivations.apply(*result)
        return Batch(phony, result.index)

    def run_invocation(self, batch: Batch, partition: ModuleWrapper, skip_trackers: List[SkipTrackerThroughPotals], invocation: Invocation) -> Batch:
        """Actually run the forward pass for a given module, and send the result
        to the next stage in the pipeline if needed."""
        task = create_task(self.checkpoint_stop, batch.index, self.group.rank(), batch, partition.module, skip_trackers)
        result = task.compute()
        task.finalize(result)
        if invocation.dest and invocation.dest.stage != invocation.this.stage:
            ranks = get_pipeline_parallel_ranks()
            dst_rank = ranks[invocation.dest.stage]
            result = self.send_async_message(dst_rank, result, invocation)
        return result

    @staticmethod
    def perform_backward_for_invocation(transport: Transport, message: PipeMessage, activations: Activations, invocation: Invocation) -> None:
        """Perform the backward pass by looking up the appropriate `Batch` and
        then calling `backward` on the tensor"""
        recvd_grads = transport.recv_message_tensors(message)
        batch: Batch = activations[invocation.this.index][invocation.order][message.args.microbatch_index]
        batch.tensor.grad_fn.grad_from_pipeline = tuple(recvd_grads.tensors)
        batch.tensor.backward(retain_graph=True)

    def run_invocations_on_batch(self, batch: Batch, invocations: Invocations, order: int, skip_trackers: List[SkipTrackerThroughPotals], activations: Activations) -> Tuple[int, int]:
        """Run invocations on the batch until we hit one that receives its input
        from a different stage (i.e. another process)"""
        invocations_handled = 0
        last_order = 0
        for invocation in invocations.values():
            if invocation.order < order:
                continue
            pi = invocation.this.index
            partition = self.partitions[pi]
            if invocation.order == order:
                invocations_handled += 1
                last_order = invocation.order
                activations[pi][invocation.order][batch.index] = self.run_invocation(batch, partition, skip_trackers, invocation)
            elif invocation.source and invocation.source.stage == self.group.rank():
                invocations_handled += 1
                last_order = invocation.order
                batch = activations[invocation.source.index][invocation.order - 1][batch.index]
                activations[pi][invocation.order][batch.index] = self.run_invocation(batch, partition, skip_trackers, invocation)
                del activations[invocation.source.index][invocation.order - 1][batch.index]
            elif invocation.source and invocation.source.stage != self.group.rank():
                break
        return (invocations_handled, last_order)

    def event_loop_head(self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals], event: Optional[Event]) -> None:
        """The event loop for the "head", which first performs the forward pass
        on any applicable layers for this stage, and then enters the common
        `event_loop_inner`"""
        invocations, activations = self.get_invocations_and_activations()
        expected_invocations = len(invocations) * len(batches)
        actual_invocations = 0
        count_per_order = dict()
        for batch in batches:
            inv_count, last_order = self.run_invocations_on_batch(batch, invocations, 0, skip_trackers, activations)
            actual_invocations += inv_count
            count_per_order[last_order] = inv_count
        if actual_invocations < expected_invocations or self.training:
            self.event_loop_inner(expected_invocations, skip_trackers, activations, invocations, count_per_order, already_received=actual_invocations, event=event)

    def get_batch_from_message(self, message: PipeMessage) -> Batch:
        """Get the tensor(s) wrapped in a `Batch` from a `PipeMessage`, applying
        AsyncRecvOperator so we can intercept the backward pass"""
        microbatch_index = message.args.microbatch_index
        phony = torch.empty(0, device=self.transport.input_device, requires_grad=True)
        result = AsyncRecvOperator.apply(phony, self.transport, message, EVENT_LOOP_QUEUE)
        if len(result) == 1:
            batch = Batch(result[0], microbatch_index)
        else:
            batch = Batch(result, microbatch_index)
        return batch

    def event_loop_tail(self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals]) -> None:
        """The event loop for the "tail", or final stage which only processes
        activations and then returns to the caller so that the loss can be
        calculated. This also handles the first/only stage for the special
        case of a 1-stage pipeline."""
        invocations, activations = self.get_invocations_and_activations()
        expected_invocations = len(invocations) * len(batches)
        actual_invocations = 0
        rank = self.group.rank()
        count_per_order = dict()
        for batch in batches:
            if rank == 0:
                order = 0
            else:
                message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
                args: AsyncMessageBody = message.args
                batch = self.get_batch_from_message(message)
                order = args.order
            inv_count, last_order = self.run_invocations_on_batch(batch, invocations, order, skip_trackers, activations)
            actual_invocations += inv_count
            count_per_order[last_order] = inv_count
            if invocations[last_order].dest is None:
                self.prepare_tail_backward(batch, activations, invocations, count_per_order, len(invocations) - inv_count)
        if actual_invocations < expected_invocations:
            expected_gradients = 0
            self.event_loop_inner(expected_invocations, skip_trackers, activations, invocations, count_per_order, already_received=actual_invocations, ignore_gradients=True, tail=True)
        _, last_invocation = invocations.popitem()
        for index, batch in activations[len(self.partitions) - 1][last_invocation.order].items():
            batches[index] = batch

    def get_invocations_and_activations(self) -> Tuple[Invocations, Activations]:
        activations: Activations = dict()
        invocations: Invocations = OrderedDict()
        for pi, partition in enumerate(self.partitions):
            activations[pi] = dict()
            for invocation in partition.invocations:
                activations[pi][invocation.order] = dict()
                invocations[invocation.order] = invocation
        invocations = OrderedDict(sorted(invocations.items(), key=lambda entry: entry[0]))
        return (invocations, activations)

    def event_loop(self, num_microbatch: int, skip_trackers: List[SkipTrackerThroughPotals]) -> None:
        """The event loop for the "middle", i.e. neither the head nor the tail"""
        invocations, activations = self.get_invocations_and_activations()
        expected_invocations = len(invocations) * num_microbatch
        self.event_loop_inner(expected_invocations, skip_trackers, activations, invocations, dict())

    def event_loop_inner(self, expected_invocations: int, skip_trackers: List[SkipTrackerThroughPotals], activations: Activations, invocations: Invocations, count_per_order: Dict[int, int], *, already_received: int=0, ignore_gradients: bool=False, event: Optional[Event]=None, tail: bool=False) -> None:
        """The common event loop shared by all stages. This processses
        activations for the forward pass, and if `self.training` is true,
        processes gradients for the backward pass."""
        num_activations = already_received
        if self.training and (not ignore_gradients):
            num_gradients = 0
        else:
            num_gradients = expected_invocations
        while num_activations < expected_invocations or num_gradients < expected_invocations:
            if num_activations == expected_invocations and num_gradients == 0 and (event is not None):
                event.wait()
            message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
            args: AsyncMessageBody = message.args
            invocation = invocations[args.order]
            if args.message_type is AsyncMessageType.Activations:
                batch = self.get_batch_from_message(message)
                inv_count, last_order = self.run_invocations_on_batch(batch, invocations, args.order, skip_trackers, activations)
                count_per_order[last_order] = inv_count
                num_activations += inv_count
                if tail and invocations[last_order].dest is None:
                    self.prepare_tail_backward(batch, activations, invocations, count_per_order, len(invocations) - inv_count)
                assert num_activations <= expected_invocations
            elif args.message_type is AsyncMessageType.Gradients:
                num_gradients += count_per_order[invocation.order]
                self.perform_backward_for_invocation(self.transport, message, activations, invocation)

    @staticmethod
    def prepare_tail_backward(batch: Batch, activations: Activations, invocations: Invocations, count_per_order: Dict[int, int], expected_gradients: int) -> None:
        if expected_gradients > 0:
            grad_fn = next((b.grad_fn for b in batch if b.requires_grad))
            assert grad_fn
            grad_fn.tail_ctx = TailBackwardContext(activations, invocations, count_per_order, expected_gradients)