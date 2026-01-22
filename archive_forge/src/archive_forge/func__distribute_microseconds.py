import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
@staticmethod
def _distribute_microseconds(todistribute, recipients, reductions):
    results = []
    remainder = todistribute
    for index, reduction in enumerate(reductions):
        additional, remainder = divmod(remainder, reduction)
        results.append(recipients[index] + additional)
    results.append(remainder)
    return tuple(results)