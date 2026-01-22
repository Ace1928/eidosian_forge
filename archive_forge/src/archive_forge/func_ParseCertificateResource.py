from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import certificate_utils
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
@staticmethod
def ParseCertificateResource(args):
    """Gets the certificate resource to be revoked based on the specified args."""
    cert_ref = args.CONCEPTS.certificate.Parse()
    if cert_ref:
        return cert_ref
    if not args.IsSpecified('issuer_pool'):
        raise exceptions.RequiredArgumentException('--issuer-pool', 'The issuing CA pool is required if a full resource name is not provided for --certificate.')
    issuer_ref = args.CONCEPTS.issuer_pool.Parse()
    if not issuer_ref:
        raise exceptions.RequiredArgumentException('--issuer-pool', "The issuer flag is not fully specified. Please add the --issuer-location flag or specify the issuer's full resource name.")
    cert_collection_name = 'privateca.projects.locations.caPools.certificates'
    if args.IsSpecified('certificate'):
        return resources.REGISTRY.Parse(args.certificate, collection=cert_collection_name, params={'projectsId': issuer_ref.projectsId, 'locationsId': issuer_ref.locationsId, 'caPoolsId': issuer_ref.caPoolsId})
    if args.IsSpecified('serial_number'):
        certificate = certificate_utils.GetCertificateBySerialNum(issuer_ref, args.serial_number)
        return resources.REGISTRY.Parse(certificate.name, collection=cert_collection_name)
    raise exceptions.OneOfArgumentsRequiredException(['--certificate', '--serial-number'], 'To revoke a Certificate, please provide either its resource ID or serial number.')