import textwrap
from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import sphinxext
def test_with_min_0(self):
    results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.IntOpt('opt_name', min=0)])))
    self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: integer\n          :Default: ``<None>``\n          :Minimum Value: 0\n        ').lstrip(), results)