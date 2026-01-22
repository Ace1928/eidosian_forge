from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddKmsKeyringResourceArgForKMS(parser, required, name):
    concept_parsers.ConceptParser.ForResource(name, GetKmsKeyRingResourceSpec(kms_prefix=False), 'The KMS keyring resource.', required=required).AddToParser(parser)