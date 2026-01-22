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
def _build_package_finder(self, options: Values, session: PipSession, target_python: Optional[TargetPython]=None, ignore_requires_python: Optional[bool]=None) -> PackageFinder:
    """
        Create a package finder appropriate to this requirement command.

        :param ignore_requires_python: Whether to ignore incompatible
            "Requires-Python" values in links. Defaults to False.
        """
    link_collector = LinkCollector.create(session, options=options)
    selection_prefs = SelectionPreferences(allow_yanked=True, format_control=options.format_control, allow_all_prereleases=options.pre, prefer_binary=options.prefer_binary, ignore_requires_python=ignore_requires_python)
    return PackageFinder.create(link_collector=link_collector, selection_prefs=selection_prefs, target_python=target_python)