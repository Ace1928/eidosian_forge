from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
def ListKeys(project, args):
    client = GetClientInstance()
    request = GetMessagesModule().KmsinventoryProjectsCryptoKeysListRequest(parent='projects/' + project)
    return list_pager.YieldFromList(client.projects_cryptoKeys, request, limit=args.limit, batch_size_attribute='pageSize', batch_size=args.page_size, field='cryptoKeys')