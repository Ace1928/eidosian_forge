from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
class _NotCalled(str):
    """
    Wrap some text with semantics

    This class is wrapped around text in order to avoid calling it.
    There are three reasons for this:

      - some instances cannot be created since they are abstract,
      - some can only be created after qApp was created,
      - some have an ugly __repr__ with angle brackets in it.

    By using derived classes, good looking instances can be created
    which can be used to generate source code or .pyi files. When the
    real object is needed, the wrapper can simply be called.
    """

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self)

    def __call__(self):
        from shibokensupport.signature.mapping import __dict__ as namespace
        text = self if self.endswith(')') else self + '()'
        return eval(text, namespace)