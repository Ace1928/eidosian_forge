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
class SimpleWheelCache(Cache):
    """A cache of wheels for future installs."""

    def __init__(self, cache_dir: str) -> None:
        super().__init__(cache_dir)

    def get_path_for_link(self, link: Link) -> str:
        """Return a directory to store cached wheels for link

        Because there are M wheels for any one sdist, we provide a directory
        to cache them in, and then consult that directory when looking up
        cache hits.

        We only insert things into the cache if they have plausible version
        numbers, so that we don't contaminate the cache with things that were
        not unique. E.g. ./package might have dozens of installs done for it
        and build a version of 0.0...and if we built and cached a wheel, we'd
        end up using the same wheel even if the source has been edited.

        :param link: The link of the sdist for which this will cache wheels.
        """
        parts = self._get_cache_path_parts(link)
        assert self.cache_dir
        return os.path.join(self.cache_dir, 'wheels', *parts)

    def get(self, link: Link, package_name: Optional[str], supported_tags: List[Tag]) -> Link:
        candidates = []
        if not package_name:
            return link
        canonical_package_name = canonicalize_name(package_name)
        for wheel_name, wheel_dir in self._get_candidates(link, canonical_package_name):
            try:
                wheel = Wheel(wheel_name)
            except InvalidWheelFilename:
                continue
            if canonicalize_name(wheel.name) != canonical_package_name:
                logger.debug('Ignoring cached wheel %s for %s as it does not match the expected distribution name %s.', wheel_name, link, package_name)
                continue
            if not wheel.supported(supported_tags):
                continue
            candidates.append((wheel.support_index_min(supported_tags), wheel_name, wheel_dir))
        if not candidates:
            return link
        _, wheel_name, wheel_dir = min(candidates)
        return Link(path_to_url(os.path.join(wheel_dir, wheel_name)))