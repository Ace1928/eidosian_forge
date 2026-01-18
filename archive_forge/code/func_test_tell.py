from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.file_part import FilePart
import gslib.tests.testcase as testcase
def test_tell(self):
    filename = 'test_tell'
    contents = 100 * b'x'
    fpath = self.CreateTempFile(file_name=filename, contents=contents)
    part_length = 23
    start_pos = 50
    fp = FilePart(fpath, start_pos, part_length)
    self.assertEqual(start_pos, fp._fp.tell())
    self.assertEqual(0, fp.tell())