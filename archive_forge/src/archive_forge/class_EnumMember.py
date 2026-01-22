import enum
import numpy as np
from .abstract import Dummy, Hashable, Literal, Number, Type
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers
class EnumMember(Type):
    """
    Type class for Enum members.
    """
    basename = 'Enum'
    class_type_class = EnumClass

    def __init__(self, cls, dtype):
        assert isinstance(cls, type)
        assert isinstance(dtype, Type)
        self.instance_class = cls
        self.dtype = dtype
        name = '%s<%s>(%s)' % (self.basename, self.dtype, self.instance_class.__name__)
        super(EnumMember, self).__init__(name)

    @property
    def key(self):
        return (self.instance_class, self.dtype)

    @property
    def class_type(self):
        """
        The type of this member's class.
        """
        return self.class_type_class(self.instance_class, self.dtype)