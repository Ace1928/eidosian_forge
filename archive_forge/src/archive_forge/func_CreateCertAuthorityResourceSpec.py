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
def CreateCertAuthorityResourceSpec(display_name, certificate_authority_attribute='certificate_authority', location_attribute='location', location_fallthroughs=None, pool_id_fallthroughs=None, ca_id_fallthroughs=None):
    return concepts.ResourceSpec('privateca.projects.locations.caPools.certificateAuthorities', api_version='v1', resource_name=display_name, certificateAuthoritiesId=CertAuthorityAttributeConfig(certificate_authority_attribute, fallthroughs=ca_id_fallthroughs), caPoolsId=CaPoolAttributeConfig(fallthroughs=pool_id_fallthroughs), locationsId=LocationAttributeConfig(location_attribute, fallthroughs=location_fallthroughs), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=True)