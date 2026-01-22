from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
def _ValidateIssuingCa(ca_pool_name, issuing_ca_id, ca_list):
    """Checks that an issuing CA is in the CA Pool and has a valid state.

  Args:
    ca_pool_name: The resource name of the containing CA Pool.
    issuing_ca_id: The CA ID of the CA to verify.
    ca_list: The list of JSON CA objects in the CA pool to check from

  Raises:
    InvalidArgumentException on validation errors
  """
    messages = privateca_base.GetMessagesModule(api_version='v1')
    allowd_issuing_states = [messages.CertificateAuthority.StateValueValuesEnum.ENABLED, messages.CertificateAuthority.StateValueValuesEnum.STAGED]
    issuing_ca = None
    for ca in ca_list:
        if 'certificateAuthorities/{}'.format(issuing_ca_id) in ca.name:
            issuing_ca = ca
    if not issuing_ca:
        raise exceptions.InvalidArgumentException('--issuer-ca', 'The specified CA with ID [{}] was not found in CA Pool [{}]'.format(issuing_ca_id, ca_pool_name))
    if issuing_ca.state not in allowd_issuing_states:
        raise exceptions.InvalidArgumentException('--issuer-pool', 'The specified CA with ID [{}] in CA Pool [{}] is not ENABLED or STAGED. Please choose a CA that has one of these states to issue the CA certificate from.'.format(issuing_ca_id, ca_pool_name))