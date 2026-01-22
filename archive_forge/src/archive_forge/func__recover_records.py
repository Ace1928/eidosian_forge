import logging
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from ..interface.interface_queue import InterfaceQueue
from ..lib import proto_util, telemetry, tracelog
from . import context, datastore, flow_control
from .settings_static import SettingsStatic
def _recover_records(self, start: int, end: int) -> None:
    sender_read = pb.SenderReadRequest(start_offset=start, final_offset=end)
    record = self._interface._make_request(sender_read=sender_read)
    self._ensure_flushed(end)
    self._forward_record(record)