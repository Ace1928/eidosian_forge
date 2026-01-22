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
class CustomVersionInfoTests(VersionInfoTestCase):

    def test_custom_null(self):
        sio = StringIO()
        wt = self.make_branch_and_tree('branch')
        builder = CustomVersionInfoBuilder(wt.branch, working_tree=wt, template='revno: {revno}')
        builder.generate(sio)
        self.assertEqual('revno: 0', sio.getvalue())
        builder = CustomVersionInfoBuilder(wt.branch, working_tree=wt, template='{revno} revid: {revision_id}')
        self.assertRaises(MissingTemplateVariable, builder.generate, sio)

    def test_custom_dotted_revno(self):
        sio = StringIO()
        wt = self.create_tree_with_dotted_revno()
        builder = CustomVersionInfoBuilder(wt.branch, working_tree=wt, template='{revno} revid: {revision_id}')
        builder.generate(sio)
        self.assertEqual('1.1.1 revid: o2', sio.getvalue())

    def regen(self, wt, tpl, **kwargs):
        sio = StringIO()
        builder = CustomVersionInfoBuilder(wt.branch, working_tree=wt, template=tpl, **kwargs)
        builder.generate(sio)
        val = sio.getvalue()
        return val

    def test_build_date(self):
        wt = self.create_branch()
        val = self.regen(wt, 'build-date: "{build_date}"\ndate: "{date}"')
        self.assertContainsRe(val, 'build-date: "[0-9-+: ]+"')
        self.assertContainsRe(val, 'date: "[0-9-+: ]+"')

    def test_revno(self):
        wt = self.create_branch()
        val = self.regen(wt, 'revno: {revno}')
        self.assertEqual(val, 'revno: 3')

    def test_revision_id(self):
        wt = self.create_branch()
        val = self.regen(wt, 'revision-id: {revision_id}')
        self.assertEqual(val, 'revision-id: r3')

    def test_clean(self):
        wt = self.create_branch()
        val = self.regen(wt, 'clean: {clean}', check_for_clean=True)
        self.assertEqual(val, 'clean: 1')

    def test_not_clean(self):
        wt = self.create_branch()
        self.build_tree(['branch/c'])
        val = self.regen(wt, 'clean: {clean}', check_for_clean=True)
        self.assertEqual(val, 'clean: 0')
        os.remove('branch/c')

    def test_custom_without_template(self):
        builder = CustomVersionInfoBuilder(None)
        sio = StringIO()
        self.assertRaises(NoTemplate, builder.generate, sio)