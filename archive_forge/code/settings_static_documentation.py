from dataclasses import fields
from typing import Any, Iterable, Sequence, Tuple
from wandb.proto import wandb_settings_pb2
from wandb.sdk.wandb_settings import SettingsData
A readonly object that wraps a protobuf Settings message.

    Implements the mapping protocol, so you can access settings as
    attributes or items.
    