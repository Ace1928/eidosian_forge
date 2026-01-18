from __future__ import annotations
from typing import Dict, TypeVar, Optional, Type, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.utils.logs import logger
def update_client_registry_mapping(mapping: Dict[str, str]):
    """
    Updates the client registry mapping
    """
    global _client_registry_mapping
    if _client_registry_mapping is None:
        _client_registry_mapping = {}
    _client_registry_mapping.update(mapping)