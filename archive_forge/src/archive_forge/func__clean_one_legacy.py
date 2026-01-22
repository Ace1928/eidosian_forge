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
def _clean_one_legacy(req: InstallRequirement, global_options: List[str]) -> bool:
    clean_args = make_setuptools_clean_args(req.setup_py_path, global_options=global_options)
    logger.info('Running setup.py clean for %s', req.name)
    try:
        call_subprocess(clean_args, command_desc='python setup.py clean', cwd=req.source_dir)
        return True
    except Exception:
        logger.error('Failed cleaning build dir for %s', req.name)
        return False