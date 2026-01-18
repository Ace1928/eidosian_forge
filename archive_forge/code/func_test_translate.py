import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_translate(self):
    for pattern, regex in TRANSLATE_TESTS:
        if re.escape(b'/') == b'/':
            regex = regex.replace(b'\\/', b'/')
        self.assertEqual(regex, translate(pattern), f'orig pattern: {pattern!r}, regex: {translate(pattern)!r}, expected: {regex!r}')