from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
def GetNotFoundMessage(conn_context, resource_ref, resource_kind='Service'):
    """Returns a user mesage for resource not found.

  Args:
    conn_context: connection_context.ConnectionInfo, Metadata for the run API
      client.
    resource_ref: protorpc.messages.Message, A resource reference object for the
      resource. See googlecloudsdk.core.resources.Registry.ParseResourceId for
      details.
    resource_kind: str, resource kind, e.g. "Service"
  """
    msg = '{resource_kind} [{resource}] could not be found in {ns_label} [{ns}] region [{region}].'
    return msg.format(resource_kind=resource_kind, resource=resource_ref.Name(), ns_label=conn_context.ns_label, ns=resource_ref.Parent().Name(), region=conn_context.region)