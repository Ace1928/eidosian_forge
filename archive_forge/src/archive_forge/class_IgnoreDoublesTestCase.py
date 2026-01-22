import io
import sys
import textwrap
from unittest import mock
import fixtures
from oslotest import base
import tempfile
import testscenarios
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_config import generator
from oslo_config import types
import yaml
class IgnoreDoublesTestCase(base.BaseTestCase):
    opts = [cfg.StrOpt('foo', help='foo option'), cfg.StrOpt('bar', help='bar option'), cfg.StrOpt('foo_bar', help='foobar'), cfg.StrOpt('str_opt', help='a string'), cfg.BoolOpt('bool_opt', help='a boolean'), cfg.IntOpt('int_opt', help='an integer')]

    def test_cleanup_opts_default(self):
        o = [('namespace1', [('group1', self.opts)])]
        self.assertEqual(o, generator._cleanup_opts(o))

    def test_cleanup_opts_dup_opt(self):
        o = [('namespace1', [('group1', self.opts + [self.opts[0]])])]
        e = [('namespace1', [('group1', self.opts)])]
        self.assertEqual(e, generator._cleanup_opts(o))

    def test_cleanup_opts_dup_groups_opt(self):
        o = [('namespace1', [('group1', self.opts + [self.opts[1]]), ('group2', self.opts), ('group3', self.opts + [self.opts[2]])])]
        e = [('namespace1', [('group1', self.opts), ('group2', self.opts), ('group3', self.opts)])]
        self.assertEqual(e, generator._cleanup_opts(o))

    def test_cleanup_opts_dup_mixed_case_groups_opt(self):
        o = [('namespace1', [('default', self.opts), ('Default', self.opts + [self.opts[1]]), ('DEFAULT', self.opts + [self.opts[2]]), ('group1', self.opts + [self.opts[1]]), ('Group1', self.opts), ('GROUP1', self.opts + [self.opts[2]])])]
        e = [('namespace1', [('DEFAULT', self.opts), ('group1', self.opts)])]
        self.assertEqual(e, generator._cleanup_opts(o))

    def test_cleanup_opts_dup_namespace_groups_opts(self):
        o = [('namespace1', [('group1', self.opts + [self.opts[1]]), ('group2', self.opts)]), ('namespace2', [('group1', self.opts + [self.opts[2]]), ('group2', self.opts)])]
        e = [('namespace1', [('group1', self.opts), ('group2', self.opts)]), ('namespace2', [('group1', self.opts), ('group2', self.opts)])]
        self.assertEqual(e, generator._cleanup_opts(o))

    @mock.patch.object(generator, '_get_raw_opts_loaders')
    def test_list_ignores_doubles(self, raw_opts_loaders):
        config_opts = [(None, [cfg.StrOpt('foo'), cfg.StrOpt('bar')])]
        raw_opts_loaders.return_value = [('namespace', lambda: config_opts), ('namespace', lambda: config_opts)]
        slurped_opts = 0
        for _, listing in generator._list_opts(['namespace']):
            for _, opts in listing:
                slurped_opts += len(opts)
        self.assertEqual(2, slurped_opts)