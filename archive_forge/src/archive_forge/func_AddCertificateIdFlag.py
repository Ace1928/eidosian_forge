from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.third_party.appengine.api import appinfo
def AddCertificateIdFlag(parser, include_no_cert):
    """Add the --certificate-id flag to a domain-mappings command."""
    certificate_id = base.Argument('--certificate-id', help='A certificate id to use for this domain. May not be used on a domain mapping with automatically managed certificates. Use the `gcloud app ssl-certificates list` to see available certificates for this app.')
    if include_no_cert:
        group = parser.add_mutually_exclusive_group()
        certificate_id.AddToParser(group)
        group.add_argument('--no-certificate-id', action='store_true', help='Do not associate any certificate with this domain.')
    else:
        certificate_id.AddToParser(parser)