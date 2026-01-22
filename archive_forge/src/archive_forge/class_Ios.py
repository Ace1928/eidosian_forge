from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
class Ios(base.Group):
    """Command group for iOS application testing."""
    detailed_help = {'DESCRIPTION': '          Explore physical iOS devices and iOS versions which are available as\n          test targets. Also run tests against your iOS app on these devices,\n          monitor your test progress, and view detailed test results in the\n          Firebase console.\n      ', 'EXAMPLES': '          To see a list of available iOS devices and supported iOS versions,\n          run:\n\n            $ {command} models list\n\n          To view information about a specific iOS model, run:\n\n            $ {command} models describe MODEL_ID\n\n          To view details about all iOS versions, run:\n\n            $ {command} versions list\n\n          To view information about a specific iOS version, run:\n\n            $ {command} versions describe VERSION_ID\n\n          To view all options available for iOS tests, run:\n\n            $ {command} run --help\n      '}

    @staticmethod
    def Args(parser):
        """Method called by Calliope to register flags common to this sub-group.

    Args:
      parser: An argparse parser used to add arguments that immediately follow
          this group in the CLI. Positional arguments are allowed.
    """

    def Filter(self, context, args):
        """Modify the context that will be given to this group's commands when run.

    Args:
      context: {str:object}, The current context, which is a set of key-value
          pairs that can be used for common initialization among commands.
      args: argparse.Namespace: The same Namespace given to the corresponding
          .Run() invocation.

    Returns:
      The refined command context.
    """
        return context