import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def _get_metadata_request(self):
    """Get requested data properties.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        request : MetadataRequest
            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` instance.
        """
    if hasattr(self, '_metadata_request'):
        requests = get_routing_for_object(self._metadata_request)
    else:
        requests = self._get_default_requests()
    return requests