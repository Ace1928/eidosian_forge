from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetClearServerTLSPolicyForHttpsProxy(name='clear-server-tls-policy'):
    """Returns the flag for clearing the Server TLS policy.

  Args:
    name: str, the name of the flag; default: 'clear-server-tls-policy'.
  """
    return base.Argument('--' + name, action='store_true', default=False, required=False, help='      Removes any attached Server TLS policy.\n      ')