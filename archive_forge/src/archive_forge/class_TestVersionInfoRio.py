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
class TestVersionInfoRio(VersionInfoTestCase):

    def test_rio_null(self):
        wt = self.make_branch_and_tree('branch')
        bio = BytesIO()
        builder = RioVersionInfoBuilder(wt.branch, working_tree=wt)
        builder.generate(bio)
        val = bio.getvalue()
        self.assertContainsRe(val, b'build-date:')
        self.assertContainsRe(val, b'revno: 0')

    def test_rio_dotted_revno(self):
        wt = self.create_tree_with_dotted_revno()
        bio = BytesIO()
        builder = RioVersionInfoBuilder(wt.branch, working_tree=wt)
        builder.generate(bio)
        val = bio.getvalue()
        self.assertContainsRe(val, b'revno: 1.1.1')

    def regen_text(self, wt, **kwargs):
        bio = BytesIO()
        builder = RioVersionInfoBuilder(wt.branch, working_tree=wt, **kwargs)
        builder.generate(bio)
        val = bio.getvalue()
        return val

    def test_simple(self):
        wt = self.create_branch()
        val = self.regen_text(wt)
        self.assertContainsRe(val, b'build-date:')
        self.assertContainsRe(val, b'date:')
        self.assertContainsRe(val, b'revno: 3')
        self.assertContainsRe(val, b'revision-id: r3')

    def test_clean(self):
        wt = self.create_branch()
        val = self.regen_text(wt, check_for_clean=True)
        self.assertContainsRe(val, b'clean: True')

    def test_no_clean(self):
        wt = self.create_branch()
        self.build_tree(['branch/c'])
        val = self.regen_text(wt, check_for_clean=True)
        self.assertContainsRe(val, b'clean: False')

    def test_history(self):
        wt = self.create_branch()
        val = self.regen_text(wt, include_revision_history=True)
        self.assertContainsRe(val, b'id: r1')
        self.assertContainsRe(val, b'message: a')
        self.assertContainsRe(val, b'id: r2')
        self.assertContainsRe(val, b'message: b')
        self.assertContainsRe(val, b'id: r3')
        self.assertContainsRe(val, b'message: \xc3\xa52')

    def regen(self, wt, **kwargs):
        bio = BytesIO()
        builder = RioVersionInfoBuilder(wt.branch, working_tree=wt, **kwargs)
        builder.generate(bio)
        bio.seek(0)
        stanzas = list(read_stanzas(bio))
        self.assertEqual(1, len(stanzas))
        return stanzas[0]

    def test_rio_version_hook(self):

        def update_stanza(rev, stanza):
            stanza.add('bla', 'bloe')
        RioVersionInfoBuilder.hooks.install_named_hook('revision', update_stanza, None)
        wt = self.create_branch()
        stanza = self.regen(wt)
        self.assertEqual(['bloe'], stanza.get_all('bla'))

    def get_one_stanza(self, stanza, key):
        new_stanzas = list(read_stanzas(BytesIO(stanza[key].encode('utf8'))))
        self.assertEqual(1, len(new_stanzas))
        return new_stanzas[0]

    def test_build_date(self):
        wt = self.create_branch()
        stanza = self.regen(wt)
        self.assertTrue('date' in stanza)
        self.assertTrue('build-date' in stanza)
        self.assertEqual(['3'], stanza.get_all('revno'))
        self.assertEqual(['r3'], stanza.get_all('revision-id'))

    def test_not_clean(self):
        wt = self.create_branch()
        self.build_tree(['branch/c'])
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True)
        self.assertEqual(['False'], stanza.get_all('clean'))

    def test_file_revisions(self):
        wt = self.create_branch()
        self.build_tree(['branch/c'])
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True)
        file_rev_stanza = self.get_one_stanza(stanza, 'file-revisions')
        self.assertEqual(['', 'a', 'b', 'c'], file_rev_stanza.get_all('path'))
        self.assertEqual(['r1', 'r3', 'r2', 'unversioned'], file_rev_stanza.get_all('revision'))

    def test_revision_history(self):
        wt = self.create_branch()
        stanza = self.regen(wt, include_revision_history=True)
        revision_stanza = self.get_one_stanza(stanza, 'revisions')
        self.assertEqual(['r1', 'r2', 'r3'], revision_stanza.get_all('id'))
        self.assertEqual(['a', 'b', 'å2'], revision_stanza.get_all('message'))
        self.assertEqual(3, len(revision_stanza.get_all('date')))

    def test_file_revisions_with_rename(self):
        wt = self.create_branch()
        self.build_tree(['branch/a', 'branch/c'])
        wt.add('c')
        wt.rename_one('b', 'd')
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True)
        file_rev_stanza = self.get_one_stanza(stanza, 'file-revisions')
        self.assertEqual(['', 'a', 'b', 'c', 'd'], file_rev_stanza.get_all('path'))
        self.assertEqual(['r1', 'modified', 'renamed to d', 'new', 'renamed from b'], file_rev_stanza.get_all('revision'))

    def test_file_revisions_with_removal(self):
        wt = self.create_branch()
        self.build_tree(['branch/a', 'branch/c'])
        wt.add('c')
        wt.rename_one('b', 'd')
        wt.commit('modified', rev_id=b'r4')
        wt.remove(['c', 'd'])
        os.remove('branch/d')
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True)
        file_rev_stanza = self.get_one_stanza(stanza, 'file-revisions')
        self.assertEqual(['', 'a', 'c', 'd'], file_rev_stanza.get_all('path'))
        self.assertEqual(['r1', 'r4', 'unversioned', 'removed'], file_rev_stanza.get_all('revision'))

    def test_revision(self):
        wt = self.create_branch()
        self.build_tree(['branch/a', 'branch/c'])
        wt.add('c')
        wt.rename_one('b', 'd')
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True, revision_id=wt.last_revision())
        file_rev_stanza = self.get_one_stanza(stanza, 'file-revisions')
        self.assertEqual(['', 'a', 'b'], file_rev_stanza.get_all('path'))
        self.assertEqual(['r1', 'r3', 'r2'], file_rev_stanza.get_all('revision'))