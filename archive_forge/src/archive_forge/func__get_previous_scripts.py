import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import (
from ..errors import RemovedConfigError
from ..warnings import SetuptoolsWarning
def _get_previous_scripts(dist: 'Distribution') -> Optional[list]:
    value = getattr(dist, 'entry_points', None) or {}
    return value.get('console_scripts')