from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

    Sets whether User Verification succeeds or fails for an authenticator.
    The default is true.

    :param authenticator_id:
    :param is_user_verified:
    