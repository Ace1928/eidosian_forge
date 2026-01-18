import datetime
from collections import deque
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from re import Pattern
from types import GeneratorType
from typing import Any, Callable, Dict, Type, Union
from uuid import UUID
from .color import Color
from .networks import NameEmail
from .types import SecretBytes, SecretStr
def timedelta_isoformat(td: datetime.timedelta) -> str:
    """
    ISO 8601 encoding for Python timedelta object.
    """
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{('-' if td.days < 0 else '')}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'