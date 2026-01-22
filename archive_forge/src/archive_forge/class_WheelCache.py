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
class WheelCache(Cache):
    """Wraps EphemWheelCache and SimpleWheelCache into a single Cache

    This Cache allows for gracefully degradation, using the ephem wheel cache
    when a certain link is not found in the simple wheel cache first.
    """

    def __init__(self, cache_dir: str) -> None:
        super().__init__(cache_dir)
        self._wheel_cache = SimpleWheelCache(cache_dir)
        self._ephem_cache = EphemWheelCache()

    def get_path_for_link(self, link: Link) -> str:
        return self._wheel_cache.get_path_for_link(link)

    def get_ephem_path_for_link(self, link: Link) -> str:
        return self._ephem_cache.get_path_for_link(link)

    def get(self, link: Link, package_name: Optional[str], supported_tags: List[Tag]) -> Link:
        cache_entry = self.get_cache_entry(link, package_name, supported_tags)
        if cache_entry is None:
            return link
        return cache_entry.link

    def get_cache_entry(self, link: Link, package_name: Optional[str], supported_tags: List[Tag]) -> Optional[CacheEntry]:
        """Returns a CacheEntry with a link to a cached item if it exists or
        None. The cache entry indicates if the item was found in the persistent
        or ephemeral cache.
        """
        retval = self._wheel_cache.get(link=link, package_name=package_name, supported_tags=supported_tags)
        if retval is not link:
            return CacheEntry(retval, persistent=True)
        retval = self._ephem_cache.get(link=link, package_name=package_name, supported_tags=supported_tags)
        if retval is not link:
            return CacheEntry(retval, persistent=False)
        return None

    @staticmethod
    def record_download_origin(cache_dir: str, download_info: DirectUrl) -> None:
        origin_path = Path(cache_dir) / ORIGIN_JSON_NAME
        if origin_path.exists():
            try:
                origin = DirectUrl.from_json(origin_path.read_text(encoding='utf-8'))
            except Exception as e:
                logger.warning('Could not read origin file %s in cache entry (%s). Will attempt to overwrite it.', origin_path, e)
            else:
                if origin.url != download_info.url:
                    logger.warning('Origin URL %s in cache entry %s does not match download URL %s. This is likely a pip bug or a cache corruption issue. Will overwrite it with the new value.', origin.url, cache_dir, download_info.url)
        origin_path.write_text(download_info.to_json(), encoding='utf-8')