import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
class MatchPatternTests(TestCase):

    def test_matches(self):
        for path, pattern in POSITIVE_MATCH_TESTS:
            self.assertTrue(match_pattern(path, pattern), f'path: {path!r}, pattern: {pattern!r}')

    def test_no_matches(self):
        for path, pattern in NEGATIVE_MATCH_TESTS:
            self.assertFalse(match_pattern(path, pattern), f'path: {path!r}, pattern: {pattern!r}')