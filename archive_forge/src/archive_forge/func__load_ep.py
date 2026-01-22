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
def _load_ep(ep: 'metadata.EntryPoint') -> Optional[Tuple[str, Type]]:
    try:
        return (ep.name, ep.load())
    except Exception as ex:
        msg = f'{ex.__class__.__name__} while trying to load entry-point {ep.name}'
        _logger.warning(f'{msg}: {ex}')
        return None