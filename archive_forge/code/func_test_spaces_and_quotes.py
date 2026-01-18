import os
import sys
import tempfile
from .. import mergetools, tests
def test_spaces_and_quotes(self):
    cmd_list = ['kdiff3', '{base}', '{this}', '{other}', '-o', '{result}']
    args, tmpfile = mergetools._subst_filename(cmd_list, 'file with "space and quotes".txt')
    self.assertEqual(['kdiff3', 'file with "space and quotes".txt.BASE', 'file with "space and quotes".txt.THIS', 'file with "space and quotes".txt.OTHER', '-o', 'file with "space and quotes".txt'], args)