from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class MonitoringBeta(base.Group):
    """Manage Cloud Monitoring dashboards and notification channels."""
    category = base.MONITORING_CATEGORY
    detailed_help = {'DESCRIPTION': '          Manage Monitoring dashboards and notification\n          channels.\n\n          More information can be found here:\n              * https://cloud.google.com/monitoring/api/v3/\n              * https://cloud.google.com/monitoring/alerts/using-channels-api\n              * https://cloud.google.com/monitoring/dashboards/api-dashboard\n      '}

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args