import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_quoted_multiline(self):
    cf = self.from_file(b'[alias]\nwho = "!who() {\\\n  git log --no-merges --pretty=format:\'%an - %ae\' $@ | uniq -c | sort -rn;\\\n};\\\nwho"\n')
    self.assertEqual(ConfigFile({(b'alias',): {b'who': b"!who() {git log --no-merges --pretty=format:'%an - %ae' $@ | uniq -c | sort -rn;};who"}}), cf)