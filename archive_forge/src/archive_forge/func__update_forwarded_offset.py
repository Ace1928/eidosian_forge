import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
def _update_forwarded_offset(self) -> None:
    self._context.last_forwarded_offset = self._context.last_written_offset