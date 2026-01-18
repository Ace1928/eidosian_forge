from __future__ import annotations
import io
import logging
import os
import re
from glob import has_magic
from pathlib import Path
from .caching import (  # noqa: F401
from .compression import compr
from .registry import filesystem, get_filesystem_class
from .utils import (
def url_to_fs(url, **kwargs):
    """
    Turn fully-qualified and potentially chained URL into filesystem instance

    Parameters
    ----------
    url : str
        The fsspec-compatible URL
    **kwargs: dict
        Extra options that make sense to a particular storage connection, e.g.
        host, port, username, password, etc.

    Returns
    -------
    filesystem : FileSystem
        The new filesystem discovered from ``url`` and created with
        ``**kwargs``.
    urlpath : str
        The file-systems-specific URL for ``url``.
    """
    known_kwargs = {'compression', 'encoding', 'errors', 'expand', 'mode', 'name_function', 'newline', 'num'}
    kwargs = {k: v for k, v in kwargs.items() if k not in known_kwargs}
    chain = _un_chain(url, kwargs)
    inkwargs = {}
    for i, ch in enumerate(reversed(chain)):
        urls, protocol, kw = ch
        if i == len(chain) - 1:
            inkwargs = dict(**kw, **inkwargs)
            continue
        inkwargs['target_options'] = dict(**kw, **inkwargs)
        inkwargs['target_protocol'] = protocol
        inkwargs['fo'] = urls
    urlpath, protocol, _ = chain[0]
    fs = filesystem(protocol, **inkwargs)
    return (fs, urlpath)