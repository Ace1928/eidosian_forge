from __future__ import annotations
import contextlib
import functools
import urllib.request
from typing import Union, List, Optional
from typing_extensions import Annotated
from pydantic import (
def validate_url_from_list(urls: Union[str, List[str]]) -> Optional[str]:
    """
    Validates between the urls and returns the first valid url
    """
    urls = [urls] if isinstance(urls, str) else urls
    return next((url for url in urls if validate_one_url(url)), None)