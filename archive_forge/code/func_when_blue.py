import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def when_blue(details):
    return details.get('color') == 'blue'