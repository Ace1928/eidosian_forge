import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
def _is_all_none(arg, *args):
    if arg is not None:
        return False
    for more_arg in args:
        if more_arg is not None:
            return False
    return True