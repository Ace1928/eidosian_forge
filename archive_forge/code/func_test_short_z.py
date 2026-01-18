import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_short_z(self):
    options, sources = parse_command_line(['-z', 'my_preimport', 'source.pyx'])
    self.assertEqual(Options.pre_import, 'my_preimport')
    self.check_default_global_options(['pre_import'])
    self.check_default_options(options)