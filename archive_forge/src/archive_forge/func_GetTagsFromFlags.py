from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import six
def GetTagsFromFlags(args, tags_message, tags_arg_name='tags'):
    """Makes the tags message object."""
    tags = getattr(args, tags_arg_name)
    if not tags:
        return None
    return tags_message(additionalProperties=[tags_message.AdditionalProperty(key=key, value=value) for key, value in sorted(six.iteritems(tags))])