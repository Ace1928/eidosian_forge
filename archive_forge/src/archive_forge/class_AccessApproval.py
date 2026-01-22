from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class AccessApproval(base.Group):
    """Manage Access Approval requests.

  Approval requests are created by Google personnel to request approval from
  Access Approval customers prior to making administrative accesses to their
  resources. Customers can act on these requests using the commands in this
  command group.
  """