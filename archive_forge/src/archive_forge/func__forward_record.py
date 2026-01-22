import logging
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from ..interface.interface_queue import InterfaceQueue
from ..lib import proto_util, telemetry, tracelog
from . import context, datastore, flow_control
from .settings_static import SettingsStatic
def _forward_record(self, record: 'pb.Record') -> None:
    self._context_keeper.add_from_record(record)
    tracelog.log_message_queue(record, self._sender_q)
    self._sender_q.put(record)