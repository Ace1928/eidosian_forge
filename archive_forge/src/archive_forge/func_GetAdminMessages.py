from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetAdminMessages():
    """Shortcut to get the latest Bigtable Admin messages."""
    return apis.GetMessagesModule('bigtableadmin', 'v2')