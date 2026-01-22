from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _ServerTlsPolicyAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='server_tls_policy', help_text='ID of the server TLS policy for {resource}.')