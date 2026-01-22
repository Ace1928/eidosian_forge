from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_security import GetClientInstance
from googlecloudsdk.api_lib.network_security import GetMessagesModule
from googlecloudsdk.core import log
def FormatSourceAddressGroup(_, arg, request):
    source_name = arg.source
    if os.path.basename(source_name) == source_name:
        location = os.path.dirname(request.addressGroup)
        request.cloneAddressGroupItemsRequest.sourceAddressGroup = '%s/%s' % (location, source_name)
    return request