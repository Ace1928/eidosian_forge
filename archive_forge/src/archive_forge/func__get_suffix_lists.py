from __future__ import annotations
import logging
import pkgutil
import re
from collections.abc import Sequence
from typing import cast
import requests
from requests_file import FileAdapter  # type: ignore[import-untyped]
from .cache import DiskCache
def _get_suffix_lists(cache: DiskCache, urls: Sequence[str], cache_fetch_timeout: float | int | None, fallback_to_snapshot: bool, session: requests.Session | None=None) -> tuple[list[str], list[str]]:
    """Fetch, parse, and cache the suffix lists."""
    try:
        text = find_first_response(cache, urls, cache_fetch_timeout=cache_fetch_timeout, session=session)
    except SuffixListNotFound as exc:
        if fallback_to_snapshot:
            maybe_pkg_data = pkgutil.get_data('tldextract', '.tld_set_snapshot')
            pkg_data = cast(bytes, maybe_pkg_data)
            text = pkg_data.decode('utf-8')
        else:
            raise exc
    public_tlds, private_tlds = extract_tlds_from_suffix_list(text)
    return (public_tlds, private_tlds)