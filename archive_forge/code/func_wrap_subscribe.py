import inspect
from collections.abc import Mapping
from functools import partial
from .argument import Argument, to_arguments
from .mountedtype import MountedType
from .resolver import default_resolver
from .structures import NonNull
from .unmountedtype import UnmountedType
from .utils import get_type
from ..utils.deprecated import warn_deprecation
def wrap_subscribe(self, parent_subscribe):
    """
        Wraps a function subscribe, using the ObjectType subscribe_{FIELD_NAME}
        (parent_subscribe) if the Field definition has no subscribe.
        """
    return parent_subscribe