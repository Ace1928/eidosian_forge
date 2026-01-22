from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddGatewayApiConfigResourceArgs(parser, verb, gateway_required=True, api_config_required=True):
    """Adds Gateway and API Config resource arguments to parser.

  Args:
    parser: parser to add arg to
    verb: action being taken with the Gateway
    gateway_required: Boolean for if Gateway is required, default is True
    api_config_required: Boolean for if API Config is required, default is True

  Returns: None
  """
    concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('gateway', GetGatewayResourceSpec(), 'Name for gateway which will be {}.'.format(verb), required=gateway_required), presentation_specs.ResourcePresentationSpec('--api-config', GetApiConfigResourceSpec(), 'Resource name for API config the gateway will use.', flag_name_overrides={'location': ''}, required=api_config_required)]).AddToParser(parser)