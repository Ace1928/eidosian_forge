from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import tags
from googlecloudsdk.api_lib.data_catalog import tags_v1
def UpdateCreateTagRequestWithInput(unused_ref, args, request):
    del unused_ref
    client = tags.TagsClient()
    return client.ParseCreateTagArgsIntoRequest(args, request)