import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
@property
def _behind_bytes(self) -> int:
    return self._context.last_forwarded_offset - self._context.last_sent_offset