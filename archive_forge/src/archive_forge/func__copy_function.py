import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
def _copy_function(deep_copy):
    if deep_copy:
        return copy.deepcopy
    else:
        return lambda x: x