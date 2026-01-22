from __future__ import absolute_import, print_function, unicode_literals
import typing
from typing import cast
import six
from copy import deepcopy
from ._typing import Text, overload
from .enums import ResourceType
from .errors import MissingInfoNamespace
from .path import join
from .permissions import Permissions
from .time import epoch_to_datetime
def _require_namespace(self, namespace):
    """Check if the given namespace is present in the info.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the given namespace is not
                present in the info.

        """
    if namespace not in self.raw:
        raise MissingInfoNamespace(namespace)