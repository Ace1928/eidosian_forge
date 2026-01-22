from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def DomainAddRegionFlag():
    """Defines a flag for adding a region."""
    return base.Argument('--add-region', help='      An additional region to provision this domain in.\n      If domain is already provisioned in region, nothing will be done in that\n      region. Supported regions are: {}.\n      '.format(', '.join(VALID_REGIONS)))