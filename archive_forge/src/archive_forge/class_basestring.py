import sys
from past.utils import with_metaclass, PY2
class basestring(with_metaclass(BaseBaseString)):
    """
    A minimal backport of the Python 2 basestring type to Py3
    """