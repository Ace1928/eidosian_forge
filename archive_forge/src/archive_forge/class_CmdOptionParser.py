from __future__ import annotations
import glob
import optparse     # pylint: disable=deprecated-module
import os
import os.path
import shlex
import sys
import textwrap
import traceback
from typing import cast, Any, NoReturn
import coverage
from coverage import Coverage
from coverage import env
from coverage.collector import HAS_CTRACER
from coverage.config import CoverageConfig
from coverage.control import DEFAULT_DATAFILE
from coverage.data import combinable_files, debug_data_file
from coverage.debug import info_header, short_stack, write_formatted_info
from coverage.exceptions import _BaseCoverageException, _ExceptionDuringRun, NoSource
from coverage.execfile import PyRunner
from coverage.results import Numbers, should_fail_under
from coverage.version import __url__
class CmdOptionParser(CoverageOptionParser):
    """Parse one of the new-style commands for coverage.py."""

    def __init__(self, action: str, options: list[optparse.Option], description: str, usage: str | None=None):
        """Create an OptionParser for a coverage.py command.

        `action` is the slug to put into `options.action`.
        `options` is a list of Option's for the command.
        `description` is the description of the command, for the help text.
        `usage` is the usage string to display in help.

        """
        if usage:
            usage = '%prog ' + usage
        super().__init__(usage=usage, description=description)
        self.set_defaults(action=action)
        self.add_options(options)
        self.cmd = action

    def __eq__(self, other: str) -> bool:
        return other == f'<CmdOptionParser:{self.cmd}>'
    __hash__ = None

    def get_prog_name(self) -> str:
        """Override of an undocumented function in optparse.OptionParser."""
        program_name = super().get_prog_name()
        return f'{program_name} {self.cmd}'