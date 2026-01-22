from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidPocoMembershipError(exceptions.Error):
    """For when the Policy Controller feature is not enabled for a membership."""