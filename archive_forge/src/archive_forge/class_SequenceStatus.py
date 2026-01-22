import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: 'SequenceStatus') -> bool:
        return status in [SequenceStatus.FINISHED_STOPPED, SequenceStatus.FINISHED_LENGTH_CAPPED, SequenceStatus.FINISHED_ABORTED, SequenceStatus.FINISHED_IGNORED]

    @staticmethod
    def get_finished_reason(status: 'SequenceStatus') -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = 'stop'
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = 'length'
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = 'abort'
        elif status == SequenceStatus.FINISHED_IGNORED:
            finish_reason = 'length'
        else:
            finish_reason = None
        return finish_reason