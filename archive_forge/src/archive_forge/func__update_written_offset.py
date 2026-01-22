import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
def _update_written_offset(self, record: 'Record') -> None:
    end_offset = record.control.end_offset
    if end_offset:
        self._context.last_written_offset = end_offset