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
def tlds(self, include_psl_private_domains: bool | None=None) -> frozenset[str]:
    """Get the currently filtered list of suffixes."""
    if include_psl_private_domains is None:
        include_psl_private_domains = self.include_psl_private_domains
    return self.tlds_incl_private if include_psl_private_domains else self.tlds_excl_private