from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import text
def AddQueryScope(indexes):
    messages = GetMessagesModule()
    scope = messages.GoogleFirestoreAdminV1Index.QueryScopeValueValuesEnum.COLLECTION
    for index in indexes:
        index.queryScope = scope
    return indexes