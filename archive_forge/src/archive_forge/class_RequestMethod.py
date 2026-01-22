import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
class RequestMethod:
    """
    A descriptor for request methods.

    .. versionadded:: 1.3

    Parameters
    ----------
    name : str
        The name of the method for which the request function should be
        created, e.g. ``"fit"`` would create a ``set_fit_request`` function.

    keys : list of str
        A list of strings which are accepted parameters by the created
        function, e.g. ``["sample_weight"]`` if the corresponding method
        accepts it as a metadata.

    validate_keys : bool, default=True
        Whether to check if the requested parameters fit the actual parameters
        of the method.

    Notes
    -----
    This class is a descriptor [1]_ and uses PEP-362 to set the signature of
    the returned function [2]_.

    References
    ----------
    .. [1] https://docs.python.org/3/howto/descriptor.html

    .. [2] https://www.python.org/dev/peps/pep-0362/
    """

    def __init__(self, name, keys, validate_keys=True):
        self.name = name
        self.keys = keys
        self.validate_keys = validate_keys

    def __get__(self, instance, owner):

        def func(**kw):
            """Updates the request for provided parameters

            This docstring is overwritten below.
            See REQUESTER_DOC for expected functionality
            """
            if not _routing_enabled():
                raise RuntimeError('This method is only available when metadata routing is enabled. You can enable it using sklearn.set_config(enable_metadata_routing=True).')
            if self.validate_keys and set(kw) - set(self.keys):
                raise TypeError(f'Unexpected args: {set(kw) - set(self.keys)}. Accepted arguments are: {set(self.keys)}')
            requests = instance._get_metadata_request()
            method_metadata_request = getattr(requests, self.name)
            for prop, alias in kw.items():
                if alias is not UNCHANGED:
                    method_metadata_request.add_request(param=prop, alias=alias)
            instance._metadata_request = requests
            return instance
        func.__name__ = f'set_{self.name}_request'
        params = [inspect.Parameter(name='self', kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=owner)]
        params.extend([inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=UNCHANGED, annotation=Optional[Union[bool, None, str]]) for k in self.keys])
        func.__signature__ = inspect.Signature(params, return_annotation=owner)
        doc = REQUESTER_DOC.format(method=self.name)
        for metadata in self.keys:
            doc += REQUESTER_DOC_PARAM.format(metadata=metadata, method=self.name)
        doc += REQUESTER_DOC_RETURN
        func.__doc__ = doc
        return func