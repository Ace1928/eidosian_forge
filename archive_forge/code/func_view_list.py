import binascii
import warnings
from webob.compat import (
@classmethod
def view_list(cls, lst):
    """
        Create a dict that is a view on the given list
        """
    if not isinstance(lst, list):
        raise TypeError('%s.view_list(obj) takes only actual list objects, not %r' % (cls.__name__, lst))
    obj = cls()
    obj._items = lst
    return obj