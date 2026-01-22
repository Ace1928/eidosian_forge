from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _GetCertificateChainPem(chains, chain_type):
    """Returns the specified certificate chain(s) from a CertChains object.

  Args:
    chains: a KeyOperationAttestation.CertChains object.
    chain_type: a string specifying the chain(s) to retrieve.

  Returns:
    A string containing the PEM-encoded certificate chain(s).

  Raises:
    exceptions.InvalidArgumentException if chain_type is not a valid chain type.
  """
    if chain_type == 'cavium':
        return ''.join(chains.caviumCerts)
    elif chain_type == 'google-card':
        return ''.join(chains.googleCardCerts)
    elif chain_type == 'google-partition':
        return ''.join(chains.googlePartitionCerts)
    elif chain_type == 'all':
        return ''.join(chains.caviumCerts + chains.googlePartitionCerts + chains.googleCardCerts)
    raise exceptions.InvalidArgumentException('{} is not a valid chain type.'.format(chain_type))