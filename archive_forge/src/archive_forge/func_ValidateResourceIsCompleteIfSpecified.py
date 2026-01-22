from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import locations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.kms import resource_args as kms_args
from googlecloudsdk.command_lib.privateca import completers as privateca_completers
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def ValidateResourceIsCompleteIfSpecified(args, resource_arg_name):
    """Raises a ParseError if the given resource_arg_name is partially specified."""
    if not hasattr(args.CONCEPTS, resource_arg_name):
        return
    concept_info = args.CONCEPTS.ArgNameToConceptInfo(resource_arg_name)
    associated_args = [util.NamespaceFormat(arg) for arg in concept_info.attribute_to_args_map.values()]
    if not [arg for arg in associated_args if args.IsSpecified(arg)]:
        return
    try:
        concept_info.ClearCache()
        concept_info.allow_empty = False
        concept_info.Parse(args)
    except concepts.InitializationError as e:
        raise handlers.ParseError(resource_arg_name, six.text_type(e))