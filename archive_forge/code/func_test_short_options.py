import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_short_options(self):
    options, sources = parse_command_line(['-V', '-l', '-+', '-t', '-v', '-v', '-v', '-p', '-D', '-a', '-3'])
    self.assertFalse(sources)
    self.assertTrue(options.show_version)
    self.assertTrue(options.use_listing_file)
    self.assertTrue(options.cplus)
    self.assertTrue(options.timestamps)
    self.assertTrue(options.verbose >= 3)
    self.assertTrue(Options.embed_pos_in_docstring)
    self.assertFalse(Options.docstrings)
    self.assertTrue(Options.annotate)
    self.assertEqual(options.language_level, 3)
    options, sources = parse_command_line(['-f', '-2', 'source.pyx'])
    self.assertTrue(sources)
    self.assertTrue(len(sources) == 1)
    self.assertFalse(options.timestamps)
    self.assertEqual(options.language_level, 2)