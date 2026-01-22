from threading import Condition
from types import TracebackType
from typing import Dict, List, Optional, Tuple, Type, Union, cast
import torch
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed import rpc
from fairscale.nn.pipe import microbatch
from fairscale.nn.pipe.checkpoint import Checkpointing, TensorOrTensors
from fairscale.nn.pipe.dependency import fork, join
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.stream import as_cuda, current_stream, is_cuda, use_device, use_stream
from fairscale.nn.pipe.worker import Task, create_workers
from .data import DataConsumer
class PartitionHandler:
    """This class processes a single partition of the pipeline.
    Args:
        module_rref: RRef to the nn.Module for this partition. It should be on the local rpc worker.
        device: The device that holds the module.
        num_inputs: Numer of inputs to the module
        num_outputs: Number of outputs of the module. If the module output is not a tuple (and it is a
            single tensor), num_outputs should be None.
        rank: The rank of the partition
        chunks: Number of micor-batches in a mini-batch
        checkpoint_stop:: Checkpointing is done only for the first checkpoint_stop chunks of a mini-batch.
    """

    def __init__(self, module_rref: rpc.RRef, device: str, num_inputs: int, num_outputs: Optional[int], rank: int, chunks: int, checkpoint_stop: int) -> None:
        self.module = module_rref.local_value()
        self.chunks = chunks
        self.device = torch.device(device)
        self.checkpoint_stop = checkpoint_stop
        self.rank = rank
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        (self.in_queue,), (self.out_queue,) = create_workers([self.device])

    def __getstate__(self) -> Dict:
        return {}

    def local_parameter_rrefs(self) -> List[rpc.RRef]:
        """
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [rpc.RRef(p) for p in self.module.parameters()]

    def make_pipeline_record(self, consumers: List[DataConsumer]) -> DistributedPipelineRecord:
        return DistributedPipelineRecord(self.device, self.rank, self.chunks, self.num_inputs, self.num_outputs, consumers)

    def run(self, pipeline_record: DistributedPipelineRecord) -> None:
        """Runs pipeline parallelism. It modifies the given batches in place."""
        m = len(pipeline_record.batches)
        self.stream = current_stream(self.device)
        for chunk in range(m):
            with record_function('feed'):
                pipeline_record.wait_for(chunk)
            pipeline_record.fence(chunk)
            self.compute(pipeline_record, chunk)
            with use_stream(self.stream):
                pipeline_record.forward_results(chunk)

    def compute(self, pipeline_record: DistributedPipelineRecord, chunk: int) -> None:
        """Runs tasks with synchronization to tensor-pipe streams."""
        checkpoint_stop = self.checkpoint_stop
        if not self.module.training:
            checkpoint_stop = 0
        exc_info: Optional[ExcInfo] = None
        batch = pipeline_record.get_batch(chunk)
        if is_cuda(self.stream):
            pipeline_record.sync_stream(chunk, as_cuda(self.stream))
        checkpoint = chunk < checkpoint_stop
        if checkpoint:

            def function(input: TensorOrTensors, chunk_id: int=chunk) -> TensorOrTensors:
                with record_function('chunk%d-rank%d' % (chunk_id, pipeline_record.rank)):
                    result = self.module(*input)
                    if self.num_outputs is None:
                        result = (result,)
                    return tuple(result)
            chk = Checkpointing(function, batch)
            task = Task(self.stream, compute=chk.checkpoint, finalize=chk.recompute)
            del function, chk
        else:

            def compute(batch: Batch=batch, chunk_id: int=chunk, rank: int=pipeline_record.rank if pipeline_record is not None else -1) -> Batch:
                with record_function('chunk%d-rank%d' % (chunk_id, pipeline_record.rank)):
                    result = self.module(*batch.tensors)
                    if self.num_outputs is None:
                        result = (result,)
                return Batch(result, chunk_id)
            task = Task(self.stream, compute=compute, finalize=None)
            del compute
        self.in_queue.put(task)
        ok, payload = self.out_queue.get()
        if exc_info is not None:
            pass
        elif not ok:
            exc_info = cast(ExcInfo, payload)
        else:
            task, batch = cast(Tuple[Task, Batch], payload)
            with use_device(self.device):
                task.finalize(batch)
            pipeline_record.batches[chunk] = batch
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    def run_pipeline(self, pipeline_record_rref: rpc.RRef) -> Optional[Tensor]:
        """Processes a min-batch on this partition.
        If this is the last partition (pipeline_record has no consumer), concatenates results of processing
        all chunks and returns the result as the output of the model on the whole mini-batch.
        """
        pipeline_record = pipeline_record_rref.local_value()
        self.run(pipeline_record)
        result: Optional[Tensor] = None
        if not pipeline_record.consumers:
            gather_result = microbatch.gather(pipeline_record.batches)
            assert len(gather_result) == 1
            result = gather_result[0]
            s0 = current_stream(result.device)
            if is_cuda(s0):
                as_cuda(s0).synchronize()
        del pipeline_record.batches
        return result