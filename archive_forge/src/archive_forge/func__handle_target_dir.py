import errno
import json
import operator
import os
import shutil
import site
from optparse import SUPPRESS_HELP, Values
from typing import List, Optional
from pip._vendor.rich import print_json
from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.req_command import (
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, InstallationError
from pip._internal.locations import get_scheme
from pip._internal.metadata import get_environment
from pip._internal.models.installation_report import InstallationReport
from pip._internal.operations.build.build_tracker import get_build_tracker
from pip._internal.operations.check import ConflictDetails, check_install_conflicts
from pip._internal.req import install_given_reqs
from pip._internal.req.req_install import (
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.filesystem import test_writable_dir
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.virtualenv import (
from pip._internal.wheel_builder import build, should_build_for_install_command
def _handle_target_dir(self, target_dir: str, target_temp_dir: TempDirectory, upgrade: bool) -> None:
    ensure_dir(target_dir)
    lib_dir_list = []
    scheme = get_scheme('', home=target_temp_dir.path)
    purelib_dir = scheme.purelib
    platlib_dir = scheme.platlib
    data_dir = scheme.data
    if os.path.exists(purelib_dir):
        lib_dir_list.append(purelib_dir)
    if os.path.exists(platlib_dir) and platlib_dir != purelib_dir:
        lib_dir_list.append(platlib_dir)
    if os.path.exists(data_dir):
        lib_dir_list.append(data_dir)
    for lib_dir in lib_dir_list:
        for item in os.listdir(lib_dir):
            if lib_dir == data_dir:
                ddir = os.path.join(data_dir, item)
                if any((s.startswith(ddir) for s in lib_dir_list[:-1])):
                    continue
            target_item_dir = os.path.join(target_dir, item)
            if os.path.exists(target_item_dir):
                if not upgrade:
                    logger.warning('Target directory %s already exists. Specify --upgrade to force replacement.', target_item_dir)
                    continue
                if os.path.islink(target_item_dir):
                    logger.warning('Target directory %s already exists and is a link. pip will not automatically replace links, please remove if replacement is desired.', target_item_dir)
                    continue
                if os.path.isdir(target_item_dir):
                    shutil.rmtree(target_item_dir)
                else:
                    os.remove(target_item_dir)
            shutil.move(os.path.join(lib_dir, item), target_item_dir)