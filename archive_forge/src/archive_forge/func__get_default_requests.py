import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
@classmethod
def _get_default_requests(cls):
    """Collect default request values.

        This method combines the information present in ``__metadata_request__*``
        class attributes, as well as determining request keys from method
        signatures.
        """
    requests = MetadataRequest(owner=cls.__name__)
    for method in SIMPLE_METHODS:
        setattr(requests, method, cls._build_request_for_signature(router=requests, method=method))
    defaults = dict()
    for base_class in reversed(inspect.getmro(cls)):
        base_defaults = {attr: value for attr, value in vars(base_class).items() if '__metadata_request__' in attr}
        defaults.update(base_defaults)
    defaults = dict(sorted(defaults.items()))
    for attr, value in defaults.items():
        substr = '__metadata_request__'
        method = attr[attr.index(substr) + len(substr):]
        for prop, alias in value.items():
            getattr(requests, method).add_request(param=prop, alias=alias)
    return requests