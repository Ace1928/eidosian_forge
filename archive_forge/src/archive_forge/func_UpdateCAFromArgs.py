from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import pem_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.util import files
def UpdateCAFromArgs(args, current_labels):
    """Creates a CA object and update mask from CA update flags.

  Requires that args has 'pem-chain' and update labels flags registered.

  Args:
    args: The parser that contains the flag values.
    current_labels: The current set of labels for the CA.

  Returns:
    A tuple with the CA object to update with and the list of strings
    representing the update mask, respectively.
  """
    messages = privateca_base.GetMessagesModule(api_version='v1')
    ca_to_update = messages.CertificateAuthority()
    update_mask = []
    if args.IsKnownAndSpecified('pem_chain'):
        ca_to_update.subordinateConfig = messages.SubordinateConfig(pemIssuerChain=messages.SubordinateConfigChain(pemCertificates=_ParsePemChainFromFile(args.pem_chain)))
        update_mask.append('subordinate_config')
    labels_diff = labels_util.Diff.FromUpdateArgs(args)
    labels_update = labels_diff.Apply(messages.CertificateAuthority.LabelsValue, current_labels)
    if labels_update.needs_update:
        ca_to_update.labels = labels_update.labels
        update_mask.append('labels')
    if not update_mask:
        raise privateca_exceptions.NoUpdateExceptions('No updates found for the requested CA.')
    return (ca_to_update, update_mask)