from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
def ValidateAndParsePemChain(pem_chain):
    """Validates and parses a pem_chain string into a list of certs.

  Args:
    pem_chain: The string represting the pem_chain.

  Returns:
    A list of the certificates that make up the chain, in the same order
    as the input.

  Raises:
    exceptions.InvalidArgumentException if the pem_chain is in an unexpected
    format.
  """
    if not re.match(_PEM_CHAIN_RE, pem_chain):
        raise exceptions.InvalidArgumentException('pem-chain', 'The pem_chain you provided was in an unexpected format.')
    certs = re.findall(_PEM_CERT_RE, pem_chain)
    for i in range(len(certs)):
        certs[i] = certs[i].strip() + '\n'
    return certs