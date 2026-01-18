from __future__ import annotations
import re
from typing import Final, Literal
from urllib.parse import urlparse
from typing_extensions import TypeAlias
Check if a string looks like an URL.

    This doesn't check if the URL is actually valid or reachable.

    Parameters
    ----------
    url : str
        The URL to check.

    allowed_schemas : Tuple[str]
        The allowed URL schemas. Default is ("http", "https").
    