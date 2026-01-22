from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.calliope.base import ReleaseTrack
def GetViolationNotificationsEnabled(violation_notifications_enabled):
    if violation_notifications_enabled.lower() == 'true':
        return True
    if violation_notifications_enabled.lower() == 'false':
        return False
    else:
        return violation_notifications_enabled