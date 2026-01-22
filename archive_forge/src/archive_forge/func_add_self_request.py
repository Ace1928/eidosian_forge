import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def add_self_request(self, obj):
    """Add `self` (as a consumer) to the routing.

        This method is used if the router is also a consumer, and hence the
        router itself needs to be included in the routing. The passed object
        can be an estimator or a
        :class:`~sklearn.utils.metadata_routing.MetadataRequest`.

        A router should add itself using this method instead of `add` since it
        should be treated differently than the other objects to which metadata
        is routed by the router.

        Parameters
        ----------
        obj : object
            This is typically the router instance, i.e. `self` in a
            ``get_metadata_routing()`` implementation. It can also be a
            ``MetadataRequest`` instance.

        Returns
        -------
        self : MetadataRouter
            Returns `self`.
        """
    if getattr(obj, '_type', None) == 'metadata_request':
        self._self_request = deepcopy(obj)
    elif hasattr(obj, '_get_metadata_request'):
        self._self_request = deepcopy(obj._get_metadata_request())
    else:
        raise ValueError('Given `obj` is neither a `MetadataRequest` nor does it implement the required API. Inheriting from `BaseEstimator` implements the required API.')
    return self