from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

    A device request opened a user prompt to select a device. Respond with the
    selectPrompt or cancelPrompt command.
    