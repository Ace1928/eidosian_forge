from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.firewall_attachments import attachment_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddAttachmentResource(release_track, parser, help_text='Path to Firewall Attachment resource'):
    """Adds Firewall attachment resource."""
    api_version = attachment_api.GetApiVersion(release_track)
    resource_spec = concepts.ResourceSpec(ATTACHMENT_RESOURCE_COLLECTION, 'firewall attachment', api_version=api_version, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=concepts.ResourceParameterAttributeConfig('zone', 'Zone of the {resource}.', parameter_name='locationsId'), firewallAttachmentsId=concepts.ResourceParameterAttributeConfig('attachment-name', 'Name of the {resource}', parameter_name='firewallAttachmentsId'))
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=ATTACHMENT_RESOURCE_NAME, concept_spec=resource_spec, required=True, group_help=help_text)
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)