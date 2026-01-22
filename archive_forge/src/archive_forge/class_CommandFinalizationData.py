from typing import (
import attr
from .parsing import (
@attr.s(auto_attribs=True)
class CommandFinalizationData:
    """Data class containing information passed to command finalization hook methods"""
    stop: bool
    statement: Optional[Statement]