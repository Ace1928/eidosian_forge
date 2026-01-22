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
def TagsFlag():
    """Makes the base.Argument for --tags flag."""
    help_parts = ['List of tags KEY=VALUE pairs to bind.', 'Each item must be expressed as', '`<tag-key-namespaced-name>=<tag-value-short-name>`.\n', 'Example: `123/environment=production,123/costCenter=marketing`\n', 'Note: Currently this field is in Preview.']
    return base.Argument('--tags', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, help='\n'.join(help_parts))