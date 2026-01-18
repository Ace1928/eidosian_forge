import textwrap
from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import sphinxext
def test_with_choices_with_descriptions(self):
    results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', choices=[('a', 'a is the best'), ('b', 'Actually, may-b I am better'), ('c', 'c, I am clearly the greatest'), (None, 'I am having none of this'), ('', '')])])))
    self.assertEqual(textwrap.dedent("\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n          :Valid Values: a, b, c, <None>, ''\n\n          .. rubric:: Possible values\n\n          a\n            a is the best\n\n          b\n            Actually, may-b I am better\n\n          c\n            c, I am clearly the greatest\n\n          <None>\n            I am having none of this\n\n          ''\n            <No description provided>\n        ").lstrip(), results)