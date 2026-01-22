import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
class _ClassInitMeta(type):
    """Metaclass for classes that can be initialized at creation time.

    Implement the method::

      @classmethod
      def _class_init(cls, new_attrs):
          pass

    on a class, and apply this metaclass to it.  The _class_init method will be
    called right after the class is created.  The 'new_attrs' param is a dict
    containing the attributes added in the definition of the class.
    """

    def __init__(cls, name, bases, attrs):
        super(_ClassInitMeta, cls).__init__(name, bases, attrs)
        cls._class_init(attrs)