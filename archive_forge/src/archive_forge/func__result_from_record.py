import json
from typing import TYPE_CHECKING, Any, Dict, Union
from wandb.proto import wandb_internal_pb2 as pb
def _result_from_record(record: 'pb.Record') -> 'pb.Result':
    result = pb.Result(uuid=record.uuid, control=record.control)
    return result