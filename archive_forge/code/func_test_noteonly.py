from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_noteonly(self):
    c = commands.NoteModifyCommand(b'foo', b'A basic note')
    self.assertEqual(b'N inline :foo\ndata 12\nA basic note', bytes(c))