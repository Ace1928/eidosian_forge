import datetime
import math
import typing as t
from wandb.util import (
@staticmethod
def types_by_class():
    if TypeRegistry._types_by_class is None:
        TypeRegistry._types_by_class = {}
    return TypeRegistry._types_by_class