from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class MonitoringAlpha(base.Group):
    """Manage Cloud Monitoring alerting policies, dashboards, notification channels, and uptime checks."""
    category = base.MONITORING_CATEGORY
    detailed_help = {'DESCRIPTION': '          Manage Monitoring alerting policies, dashboards, and notification\n          channels.\n\n          More information can be found here:\n              * https://cloud.google.com/monitoring/api/v3/\n              * https://cloud.google.com/monitoring/alerts/using-alerting-api\n              * https://cloud.google.com/monitoring/alerts/using-channels-api\n              * https://cloud.google.com/monitoring/dashboards/api-dashboard\n              * https://cloud.google.com/monitoring/uptime-checks/manage#api\n      '}

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args