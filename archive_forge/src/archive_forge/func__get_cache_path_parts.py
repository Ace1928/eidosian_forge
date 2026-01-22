import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pip._vendor.packaging.tags import Tag, interpreter_name, interpreter_version
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import InvalidWheelFilename
from pip._internal.models.direct_url import DirectUrl
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.urls import path_to_url
def _get_cache_path_parts(self, link: Link) -> List[str]:
    """Get parts of part that must be os.path.joined with cache_dir"""
    key_parts = {'url': link.url_without_fragment}
    if link.hash_name is not None and link.hash is not None:
        key_parts[link.hash_name] = link.hash
    if link.subdirectory_fragment:
        key_parts['subdirectory'] = link.subdirectory_fragment
    key_parts['interpreter_name'] = interpreter_name()
    key_parts['interpreter_version'] = interpreter_version()
    hashed = _hash_dict(key_parts)
    parts = [hashed[:2], hashed[2:4], hashed[4:6], hashed[6:]]
    return parts