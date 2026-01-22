from __future__ import annotations
import logging # isort:skip
from .. import __version__
def _base_version_helper(version: str) -> str:
    import re
    VERSION_PAT = re.compile('^(\\d+\\.\\d+\\.\\d+)((?:\\.dev|\\.rc).*)?')
    match = VERSION_PAT.search(version)
    assert match is not None
    return match.group(1)