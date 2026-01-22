import datetime
import math
import typing as t
from wandb.util import (
class TimestampType(Type):
    name = 'timestamp'
    types: t.ClassVar[t.List[type]] = [datetime.datetime, datetime.date]