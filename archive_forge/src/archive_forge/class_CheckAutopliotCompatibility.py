from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
class CheckAutopliotCompatibility(base.ListCommand):
    """Check autopilot compatibility of a running cluster."""
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '          To check autopilot compatibility of an existing cluster, run:\n\n            $ {command} sample-cluster\n          '}

    @staticmethod
    def Args(parser):
        parser.add_argument('name', help='The name of this cluster.')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        adapter = self.context['api_adapter']
        location_get = self.context['location_get']
        location = location_get(args)

        def sort_key(issue):
            return (issue.incompatibilityType, issue.constraintType)
        resp = adapter.CheckAutopilotCompatibility(adapter.ParseCluster(args.name, location))
        resp.issues = sorted(resp.issues, key=sort_key)
        self._summary = resp.summary
        return resp.issues

    def Epilog(self, results_were_displayed):
        if self._summary:
            log.out.Print('\nSummary:\n' + self._summary)