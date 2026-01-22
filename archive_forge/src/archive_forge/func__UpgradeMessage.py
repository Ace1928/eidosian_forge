from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import text
def _UpgradeMessage(field, current, new):
    from_current = 'from {}'.format(current) if current else ''
    return '{} of cluster [{}] {} will change {} to {}.'.format(node_message, name, field, from_current, new)