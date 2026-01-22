import itertools
from oslo_serialization import jsonutils
import webob
def _confirmed(cls, value):
    return cls._CONFIRMED if value else cls._INVALID