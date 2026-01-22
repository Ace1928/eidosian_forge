from __future__ import annotations
import types
from collections.abc import Hashable
from typing import (
from typing_extensions import NamedTuple  # Generic NamedTuple: Python 3.11+
from typing_extensions import OrderedDict  # Generic OrderedDict: Python 3.7.2+
from typing_extensions import Self  # Python 3.11+
from typing_extensions import TypeAlias  # Python 3.10+
from typing_extensions import Final, Protocol, runtime_checkable  # Python 3.8+
from optree import _C
from optree._C import PyTreeKind, PyTreeSpec
from optree._C import (
class _StructSequenceMeta(type):

    def __subclasscheck__(cls, subclass: type) -> bool:
        """Return whether the class is a PyStructSequence type.

        >>> import time
        >>> issubclass(time.struct_time, structseq)
        True
        >>> class MyTuple(tuple):
        ...     n_fields = 2
        ...     n_sequence_fields = 2
        ...     n_unnamed_fields = 0
        >>> issubclass(MyTuple, structseq)
        False
        """
        return is_structseq_class(subclass)

    def __instancecheck__(cls, instance: Any) -> bool:
        """Return whether the object is a PyStructSequence instance.

        >>> import sys
        >>> isinstance(sys.float_info, structseq)
        True
        >>> isinstance((1, 2), structseq)
        False
        """
        return is_structseq_instance(instance)