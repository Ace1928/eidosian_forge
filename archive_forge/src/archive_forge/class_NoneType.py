import datetime
import math
import typing as t
from wandb.util import (
class NoneType(Type):
    name = 'none'
    types: t.ClassVar[t.List[type]] = [None.__class__]