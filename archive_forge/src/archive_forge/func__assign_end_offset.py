import json
from typing import TYPE_CHECKING, Any, Dict, Union
from wandb.proto import wandb_internal_pb2 as pb
def _assign_end_offset(record: 'pb.Record', end_offset: int) -> None:
    record.control.end_offset = end_offset