import mimetypes
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.distributions.installed import InstalledDistribution
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_metadata_distribution
from pip._internal.models.direct_url import ArchiveInfo
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.network.download import BatchDownloader, Downloader
from pip._internal.network.lazy_wheel import (
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.direct_url_helpers import (
from pip._internal.utils.hashes import Hashes, MissingHashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import (
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.unpacking import unpack_file
from pip._internal.vcs import vcs
def _prepare_linked_requirement(self, req: InstallRequirement, parallel_builds: bool) -> BaseDistribution:
    assert req.link
    link = req.link
    hashes = self._get_linked_req_hashes(req)
    if hashes and req.is_wheel_from_cache:
        assert req.download_info is not None
        assert link.is_wheel
        assert link.is_file
        if isinstance(req.download_info.info, ArchiveInfo) and req.download_info.info.hashes and hashes.has_one_of(req.download_info.info.hashes):
            hashes = None
        else:
            logger.warning("The hashes of the source archive found in cache entry don't match, ignoring cached built wheel and re-downloading source.")
            req.link = req.cached_wheel_source_link
            link = req.link
    self._ensure_link_req_src_dir(req, parallel_builds)
    if link.is_existing_dir():
        local_file = None
    elif link.url not in self._downloaded:
        try:
            local_file = unpack_url(link, req.source_dir, self._download, self.verbosity, self.download_dir, hashes)
        except NetworkConnectionError as exc:
            raise InstallationError(f'Could not install requirement {req} because of HTTP error {exc} for URL {link}')
    else:
        file_path = self._downloaded[link.url]
        if hashes:
            hashes.check_against_path(file_path)
        local_file = File(file_path, content_type=None)
    if req.download_info is None:
        assert not req.editable
        req.download_info = direct_url_from_link(link, req.source_dir)
        if isinstance(req.download_info.info, ArchiveInfo) and (not req.download_info.info.hashes) and local_file:
            hash = hash_file(local_file.path)[0].hexdigest()
            req.download_info.info.hash = f'sha256={hash}'
    if local_file:
        req.local_file_path = local_file.path
    dist = _get_prepared_distribution(req, self.build_tracker, self.finder, self.build_isolation, self.check_build_deps)
    return dist