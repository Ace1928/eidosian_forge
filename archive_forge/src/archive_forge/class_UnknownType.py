import datetime
import math
import typing as t
from wandb.util import (
class UnknownType(Type):
    """An object with an unknown type.

    All assignments to an UnknownType result in the type of the assigned object except
    `None` which results in a InvalidType.
    """
    name = 'unknown'
    types: t.ClassVar[t.List[type]] = []

    def assign_type(self, wb_type: 'Type') -> 'Type':
        return wb_type if not isinstance(wb_type, NoneType) else InvalidType()