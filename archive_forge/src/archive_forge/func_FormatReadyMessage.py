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
def FormatReadyMessage(record):
    """Returns the record's status condition Ready (or equivalent) message."""
    if record.ready_condition and record.ready_condition['message']:
        symbol, color = record.ReadySymbolAndColor()
        return console_attr.GetConsoleAttr().Colorize(textwrap.fill('{} {}'.format(symbol, record.ready_condition['message']), 100), color)
    elif record.status is None:
        return console_attr.GetConsoleAttr().Colorize('Error getting status information', 'red')
    else:
        return ''