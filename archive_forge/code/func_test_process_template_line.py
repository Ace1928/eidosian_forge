import os
import re
import unittest
from distutils import debug
from distutils.log import WARN
from distutils.errors import DistutilsTemplateError
from distutils.filelist import glob_to_re, translate_pattern, FileList
from distutils import filelist
from test.support import os_helper
from test.support import captured_stdout
from distutils.tests import support
def test_process_template_line(self):
    file_list = FileList()
    l = make_local_path
    file_list.allfiles = ['foo.tmp', 'ok', 'xo', 'four.txt', 'buildout.cfg', l('.hg/last-message.txt'), l('global/one.txt'), l('global/two.txt'), l('global/files.x'), l('global/here.tmp'), l('f/o/f.oo'), l('dir/graft-one'), l('dir/dir2/graft2'), l('dir3/ok'), l('dir3/sub/ok.txt')]
    for line in MANIFEST_IN.split('\n'):
        if line.strip() == '':
            continue
        file_list.process_template_line(line)
    wanted = ['ok', 'buildout.cfg', 'four.txt', l('.hg/last-message.txt'), l('global/one.txt'), l('global/two.txt'), l('f/o/f.oo'), l('dir/graft-one'), l('dir/dir2/graft2')]
    self.assertEqual(file_list.files, wanted)