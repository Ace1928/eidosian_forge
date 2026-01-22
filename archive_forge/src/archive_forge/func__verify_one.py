import logging
import os.path
import re
import shutil
from typing import Iterable, List, Optional, Tuple
from pip._vendor.packaging.utils import canonicalize_name, canonicalize_version
from pip._vendor.packaging.version import InvalidVersion, Version
from pip._internal.cache import WheelCache
from pip._internal.exceptions import InvalidWheelFilename, UnsupportedWheel
from pip._internal.metadata import FilesystemWheel, get_wheel_distribution
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.build.wheel import build_wheel_pep517
from pip._internal.operations.build.wheel_editable import build_wheel_editable
from pip._internal.operations.build.wheel_legacy import build_wheel_legacy
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import ensure_dir, hash_file
from pip._internal.utils.setuptools_build import make_setuptools_clean_args
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import vcs
def _verify_one(req: InstallRequirement, wheel_path: str) -> None:
    canonical_name = canonicalize_name(req.name or '')
    w = Wheel(os.path.basename(wheel_path))
    if canonicalize_name(w.name) != canonical_name:
        raise InvalidWheelFilename(f'Wheel has unexpected file name: expected {canonical_name!r}, got {w.name!r}')
    dist = get_wheel_distribution(FilesystemWheel(wheel_path), canonical_name)
    dist_verstr = str(dist.version)
    if canonicalize_version(dist_verstr) != canonicalize_version(w.version):
        raise InvalidWheelFilename(f'Wheel has unexpected file name: expected {dist_verstr!r}, got {w.version!r}')
    metadata_version_value = dist.metadata_version
    if metadata_version_value is None:
        raise UnsupportedWheel('Missing Metadata-Version')
    try:
        metadata_version = Version(metadata_version_value)
    except InvalidVersion:
        msg = f'Invalid Metadata-Version: {metadata_version_value}'
        raise UnsupportedWheel(msg)
    if metadata_version >= Version('1.2') and (not isinstance(dist.version, Version)):
        raise UnsupportedWheel(f'Metadata 1.2 mandates PEP 440 version, but {dist_verstr!r} is not')