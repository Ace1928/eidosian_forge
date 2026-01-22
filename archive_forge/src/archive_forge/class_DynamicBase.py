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
class DynamicBase(Base):
    """Base type for complex types for which all attributes are not
    defined when the class is constructed.

    This class is meant to be used as a base for types that have
    properties added after the main class is created, such as by
    loading plugins.

    """

    @classmethod
    def add_attributes(cls, **attrs):
        """Add more attributes

        The arguments should be valid Python attribute names
        associated with a type for the new attribute.

        """
        for n, t in attrs.items():
            setattr(cls, n, t)
        cls.__registry__.reregister(cls)