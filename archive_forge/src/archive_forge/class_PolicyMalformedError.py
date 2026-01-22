from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
import six
class PolicyMalformedError(PolicyError):
    """Raised when the specified Ops Agents policy is malformed."""

    def __init__(self, policy_id):
        message = 'Encountered a malformed policy. The Ops Agents policy [{policy_id}] may have been modified directly by the OS Config guest policy API / gcloud commands. If so, please delete and re-create with the Ops Agents policy gcloud commands. If not, this may be an internal error.'.format(policy_id=policy_id)
        super(PolicyMalformedError, self).__init__(message)