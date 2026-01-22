import base64
import datetime
import decimal
import inspect
import io
import logging
import netaddr
import re
import sys
import uuid
import weakref
from wsme import exc
class wsproperty(property):
    """
    A specialised :class:`property` to define typed-property on complex types.
    Example::

        class MyComplexType(wsme.types.Base):
            def get_aint(self):
                return self._aint

            def set_aint(self, value):
                assert avalue < 10  # Dummy input validation
                self._aint = value

            aint = wsproperty(int, get_aint, set_aint, mandatory=True)
    """

    def __init__(self, datatype, fget, fset=None, mandatory=False, doc=None, name=None):
        property.__init__(self, fget, fset)
        self.key = None
        self.name = name
        self.datatype = datatype
        self.mandatory = mandatory