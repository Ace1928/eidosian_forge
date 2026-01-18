import logging
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from ..interface.interface_queue import InterfaceQueue
from ..lib import proto_util, telemetry, tracelog
from . import context, datastore, flow_control
from .settings_static import SettingsStatic
def write_request_status_report(self, record: 'pb.Record') -> None:
    self._status_report = record.request.status_report
    self._write(record)