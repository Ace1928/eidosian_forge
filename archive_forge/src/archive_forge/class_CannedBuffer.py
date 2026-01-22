import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
class CannedBuffer(CannedBytes):
    """A canned buffer."""
    wrap = buffer