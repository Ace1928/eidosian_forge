import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np
class NumPyRandomBitGeneratorType(Type):

    def __init__(self, *args, **kwargs):
        super(NumPyRandomBitGeneratorType, self).__init__(*args, **kwargs)
        self.name = 'NumPyRandomBitGeneratorType'