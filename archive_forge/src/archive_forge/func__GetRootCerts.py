from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.privateca import pem_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _GetRootCerts(self, ca_pool_ref):
    """Returns the root CA certs for all active CAs in the CA pool."""
    client = privateca_base.GetClientInstance('v1')
    messages = privateca_base.GetMessagesModule('v1')
    fetch_ca_certs_response = client.projects_locations_caPools.FetchCaCerts(messages.PrivatecaProjectsLocationsCaPoolsFetchCaCertsRequest(caPool=ca_pool_ref.RelativeName(), fetchCaCertsRequest=messages.FetchCaCertsRequest(requestId=request_utils.GenerateRequestId())))
    root_certs = [chain.certificates[-1] for chain in fetch_ca_certs_response.caCerts]
    return ''.join(pem_utils.PemChainForOutput(root_certs))