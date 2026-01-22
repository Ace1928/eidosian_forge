from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Uptime(base.Group):
    """Manage Cloud Monitoring uptime checks and synthetic monitors."""
    detailed_help = {'DESCRIPTION': '          Manage Monitoring uptime checks and synthetic monitors.\n\n          More information can be found here:\n          https://cloud.google.com/monitoring/api/v3/\n      '}