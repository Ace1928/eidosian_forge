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
def test_process_template(self):
    l = make_local_path
    file_list = FileList()
    for action in ('include', 'exclude', 'global-include', 'global-exclude', 'recursive-include', 'recursive-exclude', 'graft', 'prune', 'blarg'):
        self.assertRaises(DistutilsTemplateError, file_list.process_template_line, action)
    file_list = FileList()
    file_list.set_allfiles(['a.py', 'b.txt', l('d/c.py')])
    file_list.process_template_line('include *.py')
    self.assertEqual(file_list.files, ['a.py'])
    self.assertNoWarnings()
    file_list.process_template_line('include *.rb')
    self.assertEqual(file_list.files, ['a.py'])
    self.assertWarnings()
    file_list = FileList()
    file_list.files = ['a.py', 'b.txt', l('d/c.py')]
    file_list.process_template_line('exclude *.py')
    self.assertEqual(file_list.files, ['b.txt', l('d/c.py')])
    self.assertNoWarnings()
    file_list.process_template_line('exclude *.rb')
    self.assertEqual(file_list.files, ['b.txt', l('d/c.py')])
    self.assertWarnings()
    file_list = FileList()
    file_list.set_allfiles(['a.py', 'b.txt', l('d/c.py')])
    file_list.process_template_line('global-include *.py')
    self.assertEqual(file_list.files, ['a.py', l('d/c.py')])
    self.assertNoWarnings()
    file_list.process_template_line('global-include *.rb')
    self.assertEqual(file_list.files, ['a.py', l('d/c.py')])
    self.assertWarnings()
    file_list = FileList()
    file_list.files = ['a.py', 'b.txt', l('d/c.py')]
    file_list.process_template_line('global-exclude *.py')
    self.assertEqual(file_list.files, ['b.txt'])
    self.assertNoWarnings()
    file_list.process_template_line('global-exclude *.rb')
    self.assertEqual(file_list.files, ['b.txt'])
    self.assertWarnings()
    file_list = FileList()
    file_list.set_allfiles(['a.py', l('d/b.py'), l('d/c.txt'), l('d/d/e.py')])
    file_list.process_template_line('recursive-include d *.py')
    self.assertEqual(file_list.files, [l('d/b.py'), l('d/d/e.py')])
    self.assertNoWarnings()
    file_list.process_template_line('recursive-include e *.py')
    self.assertEqual(file_list.files, [l('d/b.py'), l('d/d/e.py')])
    self.assertWarnings()
    file_list = FileList()
    file_list.files = ['a.py', l('d/b.py'), l('d/c.txt'), l('d/d/e.py')]
    file_list.process_template_line('recursive-exclude d *.py')
    self.assertEqual(file_list.files, ['a.py', l('d/c.txt')])
    self.assertNoWarnings()
    file_list.process_template_line('recursive-exclude e *.py')
    self.assertEqual(file_list.files, ['a.py', l('d/c.txt')])
    self.assertWarnings()
    file_list = FileList()
    file_list.set_allfiles(['a.py', l('d/b.py'), l('d/d/e.py'), l('f/f.py')])
    file_list.process_template_line('graft d')
    self.assertEqual(file_list.files, [l('d/b.py'), l('d/d/e.py')])
    self.assertNoWarnings()
    file_list.process_template_line('graft e')
    self.assertEqual(file_list.files, [l('d/b.py'), l('d/d/e.py')])
    self.assertWarnings()
    file_list = FileList()
    file_list.files = ['a.py', l('d/b.py'), l('d/d/e.py'), l('f/f.py')]
    file_list.process_template_line('prune d')
    self.assertEqual(file_list.files, ['a.py', l('f/f.py')])
    self.assertNoWarnings()
    file_list.process_template_line('prune e')
    self.assertEqual(file_list.files, ['a.py', l('f/f.py')])
    self.assertWarnings()