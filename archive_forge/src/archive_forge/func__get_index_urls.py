import logging
import os
import sys
from functools import partial
from optparse import Values
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.command_context import CommandContextMixIn
from pip._internal.exceptions import CommandError, PreviousBuildDirError
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
from pip._internal.req.req_file import parse_requirements
from pip._internal.req.req_install import InstallRequirement
from pip._internal.resolution.base import BaseResolver
from pip._internal.self_outdated_check import pip_self_version_check
from pip._internal.utils.temp_dir import (
from pip._internal.utils.virtualenv import running_under_virtualenv
@classmethod
def _get_index_urls(cls, options: Values) -> Optional[List[str]]:
    """Return a list of index urls from user-provided options."""
    index_urls = []
    if not getattr(options, 'no_index', False):
        url = getattr(options, 'index_url', None)
        if url:
            index_urls.append(url)
    urls = getattr(options, 'extra_index_urls', None)
    if urls:
        index_urls.extend(urls)
    return index_urls or None