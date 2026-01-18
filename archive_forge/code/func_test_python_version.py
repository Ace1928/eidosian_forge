import os
import re
from io import BytesIO, StringIO
import yaml
from .. import registry, tests, version_info_formats
from ..bzr.rio import read_stanzas
from ..version_info_formats.format_custom import (CustomVersionInfoBuilder,
from ..version_info_formats.format_python import PythonVersionInfoBuilder
from ..version_info_formats.format_rio import RioVersionInfoBuilder
from ..version_info_formats.format_yaml import YamlVersionInfoBuilder
from . import TestCaseWithTransport
def test_python_version(self):
    wt = self.create_branch()
    tvi = self.regen(wt)
    self.assertEqual('3', tvi['version_info']['revno'])
    self.assertEqual(b'r3', tvi['version_info']['revision_id'])
    self.assertTrue('date' in tvi['version_info'])
    self.assertEqual(None, tvi['version_info']['clean'])
    tvi = self.regen(wt, check_for_clean=True)
    self.assertTrue(tvi['version_info']['clean'])
    self.build_tree(['branch/c'])
    tvi = self.regen(wt, check_for_clean=True, include_file_revisions=True)
    self.assertFalse(tvi['version_info']['clean'])
    self.assertEqual(['', 'a', 'b', 'c'], sorted(tvi['file_revisions'].keys()))
    self.assertEqual(b'r3', tvi['file_revisions']['a'])
    self.assertEqual(b'r2', tvi['file_revisions']['b'])
    self.assertEqual('unversioned', tvi['file_revisions']['c'])
    os.remove('branch/c')
    tvi = self.regen(wt, include_revision_history=True)
    rev_info = [(rev, message) for rev, message, timestamp, timezone in tvi['revisions']]
    self.assertEqual([(b'r1', 'a'), (b'r2', 'b'), (b'r3', 'å2')], rev_info)
    self.build_tree(['branch/a', 'branch/c'])
    wt.add('c')
    wt.rename_one('b', 'd')
    tvi = self.regen(wt, check_for_clean=True, include_file_revisions=True)
    self.assertEqual(['', 'a', 'b', 'c', 'd'], sorted(tvi['file_revisions'].keys()))
    self.assertEqual('modified', tvi['file_revisions']['a'])
    self.assertEqual('renamed to d', tvi['file_revisions']['b'])
    self.assertEqual('new', tvi['file_revisions']['c'])
    self.assertEqual('renamed from b', tvi['file_revisions']['d'])
    wt.commit('modified', rev_id=b'r4')
    wt.remove(['c', 'd'])
    os.remove('branch/d')
    tvi = self.regen(wt, check_for_clean=True, include_file_revisions=True)
    self.assertEqual(['', 'a', 'c', 'd'], sorted(tvi['file_revisions'].keys()))
    self.assertEqual(b'r4', tvi['file_revisions']['a'])
    self.assertEqual('unversioned', tvi['file_revisions']['c'])
    self.assertEqual('removed', tvi['file_revisions']['d'])