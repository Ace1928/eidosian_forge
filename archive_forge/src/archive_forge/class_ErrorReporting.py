from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class ErrorReporting(base.Group):
    """Manage Stackdriver Error Reporting."""
    category = base.MONITORING_CATEGORY

    def Filter(self, context, args):
        """Modify the context that will be given to this group's commands when run.

    Args:
      context: The current context.
      args: The argparse namespace given to the corresponding .Run() invocation.

    Returns:
      The updated context.
    """
        base.RequireProjectID(args)
        base.DisableUserProjectQuota()
        context['clouderrorreporting_client_v1beta1'] = apis.GetClientInstance('clouderrorreporting', 'v1beta1')
        context['clouderrorreporting_messages_v1beta1'] = apis.GetMessagesModule('clouderrorreporting', 'v1beta1')
        context['clouderrorreporting_resources'] = resources
        return context