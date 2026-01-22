from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import textwrap
from typing import Mapping
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def BuildHeader(record):
    con = console_attr.GetConsoleAttr()
    status = con.Colorize(*record.ReadySymbolAndColor())
    try:
        place = 'region ' + record.region
    except KeyError:
        place = 'namespace ' + record.namespace
    return con.Emphasize('{} {} {} in {}'.format(status, record.Kind(), record.name, place))