from __future__ import annotations
import logging
import os
import urllib.parse
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import wraps
import idna
import requests
from .cache import DiskCache, get_cache_dir
from .remote import lenient_netloc, looks_like_ip, looks_like_ipv6
from .suffix_list import get_suffix_lists
def _get_tld_extractor(self, session: requests.Session | None=None) -> _PublicSuffixListTLDExtractor:
    """Get or compute this object's TLDExtractor.

        Looks up the TLDExtractor in roughly the following order, based on the
        settings passed to __init__:

        1. Memoized on `self`
        2. Local system _cache file
        3. Remote PSL, over HTTP
        4. Bundled PSL snapshot file
        """
    if self._extractor:
        return self._extractor
    public_tlds, private_tlds = get_suffix_lists(cache=self._cache, urls=self.suffix_list_urls, cache_fetch_timeout=self.cache_fetch_timeout, fallback_to_snapshot=self.fallback_to_snapshot, session=session)
    if not any([public_tlds, private_tlds, self.extra_suffixes]):
        raise ValueError('No tlds set. Cannot proceed without tlds.')
    self._extractor = _PublicSuffixListTLDExtractor(public_tlds=public_tlds, private_tlds=private_tlds, extra_tlds=list(self.extra_suffixes), include_psl_private_domains=self.include_psl_private_domains)
    return self._extractor