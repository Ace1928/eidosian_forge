from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def UpdateIdentifyingFields(ref, args, request):
    """Update bigQueryOptions.identifyingFields with parsed fields."""
    del ref
    big_query_options = request.googlePrivacyDlpV2CreateDlpJobRequest.inspectJob.storageConfig.bigQueryOptions
    if big_query_options and args.identifying_fields:
        field_id = _GetMessageClass('GooglePrivacyDlpV2FieldId')
        big_query_options.identifyingFields = [field_id(name=field) for field in args.identifying_fields]
    return request