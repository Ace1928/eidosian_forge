from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
def GetApiMessages():
    return core_apis.GetMessagesModule('replicapoolupdater', 'v1beta1')