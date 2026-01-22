import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def _version_str_to_list(version: str) -> List[int]:
    """convert a version string to a list of ints

    non-int segments are excluded
    """
    v = []
    for part in version.split('.'):
        try:
            v.append(int(part))
        except ValueError:
            pass
    return v