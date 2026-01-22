from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
def ParseCreateTagTemplateArgsIntoRequest(self, args, request):
    """Parses tag-templates create args into the request."""
    fields = []
    for field in args.field:
        fields.append(self._ParseField(field))
    arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1TagTemplate.fields', self.messages.GoogleCloudDatacatalogV1TagTemplate.FieldsValue(additionalProperties=fields))
    return request