import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_warning_extra_dont_overwrite(self):
    options, sources = parse_command_line(['-X', 'cdivision=True', '--warning-extra', '-X', 'c_string_type=bytes', 'source.pyx'])
    self.assertTrue(len(options.compiler_directives), len(Options.extra_warnings) + 1)
    self.check_default_global_options()
    self.check_default_options(options, ['compiler_directives'])