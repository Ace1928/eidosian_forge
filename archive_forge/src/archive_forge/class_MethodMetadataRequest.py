import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
class MethodMetadataRequest:
    """A prescription of how metadata is to be passed to a single method.

    Refer to :class:`MetadataRequest` for how this class is used.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        A display name for the object owning these requests.

    method : str
        The name of the method to which these requests belong.

    requests : dict of {str: bool, None or str}, default=None
        The initial requests for this method.
    """

    def __init__(self, owner, method, requests=None):
        self._requests = requests or dict()
        self.owner = owner
        self.method = method

    @property
    def requests(self):
        """Dictionary of the form: ``{key: alias}``."""
        return self._requests

    def add_request(self, *, param, alias):
        """Add request info for a metadata.

        Parameters
        ----------
        param : str
            The property for which a request is set.

        alias : str, or {True, False, None}
            Specifies which metadata should be routed to `param`

            - str: the name (or alias) of metadata given to a meta-estimator that
              should be routed to this parameter.

            - True: requested

            - False: not requested

            - None: error if passed
        """
        if not request_is_alias(alias) and (not request_is_valid(alias)):
            raise ValueError(f"The alias you're setting for `{param}` should be either a valid identifier or one of {{None, True, False}}, but given value is: `{alias}`")
        if alias == param:
            alias = True
        if alias == UNUSED:
            if param in self._requests:
                del self._requests[param]
            else:
                raise ValueError(f"Trying to remove parameter {param} with UNUSED which doesn't exist.")
        else:
            self._requests[param] = alias
        return self

    def _get_param_names(self, return_alias):
        """Get names of all metadata that can be consumed or routed by this method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        return_alias : bool
            Controls whether original or aliased names should be returned. If
            ``False``, aliases are ignored and original names are returned.

        Returns
        -------
        names : set of str
            A set of strings with the names of all parameters.
        """
        return set((alias if return_alias and (not request_is_valid(alias)) else prop for prop, alias in self._requests.items() if not request_is_valid(alias) or alias is not False))

    def _check_warnings(self, *, params):
        """Check whether metadata is passed which is marked as WARN.

        If any metadata is passed which is marked as WARN, a warning is raised.

        Parameters
        ----------
        params : dict
            The metadata passed to a method.
        """
        params = {} if params is None else params
        warn_params = {prop for prop, alias in self._requests.items() if alias == WARN and prop in params}
        for param in warn_params:
            warn(f'Support for {param} has recently been added to this class. To maintain backward compatibility, it is ignored now. You can set the request value to False to silence this warning, or to True to consume and use the metadata.')

    def _route_params(self, params):
        """Prepare the given parameters to be passed to the method.

        The output of this method can be used directly as the input to the
        corresponding method as extra props.

        Parameters
        ----------
        params : dict
            A dictionary of provided metadata.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to the
            corresponding method.
        """
        self._check_warnings(params=params)
        unrequested = dict()
        args = {arg: value for arg, value in params.items() if value is not None}
        res = Bunch()
        for prop, alias in self._requests.items():
            if alias is False or alias == WARN:
                continue
            elif alias is True and prop in args:
                res[prop] = args[prop]
            elif alias is None and prop in args:
                unrequested[prop] = args[prop]
            elif alias in args:
                res[prop] = args[alias]
        if unrequested:
            raise UnsetMetadataPassedError(message=f'[{', '.join([key for key in unrequested])}] are passed but are not explicitly set as requested or not for {self.owner}.{self.method}', unrequested_params=unrequested, routed_params=res)
        return res

    def _consumes(self, params):
        """Check whether the given parameters are consumed by this method.

        Parameters
        ----------
        params : iterable of str
            An iterable of parameters to check.

        Returns
        -------
        consumed : set of str
            A set of parameters which are consumed by this method.
        """
        params = set(params)
        res = set()
        for prop, alias in self._requests.items():
            if alias is True and prop in params:
                res.add(prop)
            elif isinstance(alias, str) and alias in params:
                res.add(alias)
        return res

    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : dict
            A serialized version of the instance in the form of a dictionary.
        """
        return self._requests

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))