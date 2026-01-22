import abc
import json
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
@classmethod
def _extract_kwargs(cls, kwargs):
    """Remove parameters related to this method from other kwargs."""
    return dict([(p, kwargs.pop(p, None)) for p in cls._method_parameters])