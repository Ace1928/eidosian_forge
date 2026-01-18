import textwrap
from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import sphinxext
def test_none_in_default(self):
    results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', help='this appears in the default group')])))
    self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n\n          this appears in the default group\n        ').lstrip(), results)