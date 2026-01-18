import os
import sys
import tempfile
import textwrap
import shutil
import subprocess
import unittest
from traits.api import (
from traits.testing.optional_dependencies import requires_numpy
def test_symbol_deprecated(self):
    with self.assertWarnsRegex(DeprecationWarning, 'Symbol trait type'):
        Symbol('random:random')