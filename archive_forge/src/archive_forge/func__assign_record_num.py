import json
from typing import TYPE_CHECKING, Any, Dict, Union
from wandb.proto import wandb_internal_pb2 as pb
def _assign_record_num(record: 'pb.Record', record_num: int) -> None:
    record.num = record_num