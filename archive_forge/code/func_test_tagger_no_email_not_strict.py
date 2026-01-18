import io
import time
import unittest
from fastimport import (
from :2
def test_tagger_no_email_not_strict(self):
    p = parser.ImportParser(io.BytesIO(b'tag refs/tags/v1.0\nfrom :xxx\ntagger Joe Wong\ndata 11\ncreate v1.0'), strict=False)
    cmds = list(p.iter_commands())
    self.assertEqual(1, len(cmds))
    self.assertTrue(isinstance(cmds[0], commands.TagCommand))
    self.assertEqual(cmds[0].tagger[:2], (b'Joe Wong', None))