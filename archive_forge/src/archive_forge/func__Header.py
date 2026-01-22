from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_property
def _Header(self, record):
    con = console_attr.GetConsoleAttr()
    return con.Emphasize('Component {}'.format(record['metadata']['name']))