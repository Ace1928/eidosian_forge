import logging
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from ..interface.interface_queue import InterfaceQueue
from ..lib import proto_util, telemetry, tracelog
from . import context, datastore, flow_control
from .settings_static import SettingsStatic
def write_request(self, record: 'pb.Record') -> None:
    request_type = record.request.WhichOneof('request_type')
    assert request_type
    write_request_str = 'write_request_' + request_type
    write_request_handler: Optional[Callable[[pb.Record], None]] = getattr(self, write_request_str, None)
    if write_request_handler:
        return write_request_handler(record)
    self._write(record)