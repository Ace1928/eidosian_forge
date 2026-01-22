import textwrap
from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import sphinxext
class FormatGroupTest(base.BaseTestCase):

    def test_none_in_default(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', help='this appears in the default group')])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n\n          this appears in the default group\n        ').lstrip(), results)

    def test_with_default_value(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', default='this is the default', help='this appears in the default group')])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``this is the default``\n\n          this appears in the default group\n        ').lstrip(), results)

    def test_with_min(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.IntOpt('opt_name', min=1)])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: integer\n          :Default: ``<None>``\n          :Minimum Value: 1\n        ').lstrip(), results)

    def test_with_min_0(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.IntOpt('opt_name', min=0)])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: integer\n          :Default: ``<None>``\n          :Minimum Value: 0\n        ').lstrip(), results)

    def test_with_max(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.IntOpt('opt_name', max=1)])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: integer\n          :Default: ``<None>``\n          :Maximum Value: 1\n        ').lstrip(), results)

    def test_with_max_0(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.IntOpt('opt_name', max=0)])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: integer\n          :Default: ``<None>``\n          :Maximum Value: 0\n        ').lstrip(), results)

    def test_with_choices(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', choices=['a', 'b', 'c', None, ''])])))
        self.assertEqual(textwrap.dedent("\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n          :Valid Values: a, b, c, <None>, ''\n        ").lstrip(), results)

    def test_with_choices_with_descriptions(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', choices=[('a', 'a is the best'), ('b', 'Actually, may-b I am better'), ('c', 'c, I am clearly the greatest'), (None, 'I am having none of this'), ('', '')])])))
        self.assertEqual(textwrap.dedent("\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n          :Valid Values: a, b, c, <None>, ''\n\n          .. rubric:: Possible values\n\n          a\n            a is the best\n\n          b\n            Actually, may-b I am better\n\n          c\n            c, I am clearly the greatest\n\n          <None>\n            I am having none of this\n\n          ''\n            <No description provided>\n        ").lstrip(), results)

    def test_group_obj_without_help(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name='group', group_obj=cfg.OptGroup('group'), opt_list=[cfg.StrOpt('opt_name')])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: group\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n        ').lstrip(), results)

    def test_group_obj_with_help(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name='group', group_obj=cfg.OptGroup('group', help='group help'), opt_list=[cfg.StrOpt('opt_name')])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: group\n\n          group help\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n        ').lstrip(), results)

    def test_deprecated_opts_without_deprecated_group(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', deprecated_name='deprecated_name')])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n\n          .. list-table:: Deprecated Variations\n             :header-rows: 1\n\n             - * Group\n               * Name\n             - * DEFAULT\n               * deprecated_name\n        ').lstrip(), results)

    def test_deprecated_opts_with_deprecated_group(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', deprecated_name='deprecated_name', deprecated_group='deprecated_group')])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n\n          .. list-table:: Deprecated Variations\n             :header-rows: 1\n\n             - * Group\n               * Name\n             - * deprecated_group\n               * deprecated_name\n        ').lstrip(), results)

    def test_deprecated_for_removal(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', deprecated_for_removal=True, deprecated_reason='because I said so', deprecated_since='13.0')])))
        self.assertIn('.. warning::', results)
        self.assertIn('because I said so', results)
        self.assertIn('since 13.0', results)

    def test_mutable(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.IntOpt('opt_name', mutable=True)])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: integer\n          :Default: ``<None>``\n          :Mutable: This option can be changed without restarting.\n        ').lstrip(), results)

    def test_not_mutable(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.IntOpt('opt_name', mutable=False)])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: integer\n          :Default: ``<None>``\n        ').lstrip(), results)

    def test_advanced(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', advanced=True)])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n          :Advanced Option: Intended for advanced users and not used\n              by the majority of users, and might have a significant\n              effect on stability and/or performance.\n        ').lstrip(), results)

    def test_not_advanced(self):
        results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', advanced=False)])))
        self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n        ').lstrip(), results)