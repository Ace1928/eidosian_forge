from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Android(base.Group):
    """Command group for Android application testing."""
    detailed_help = {'DESCRIPTION': '          Explore physical and virtual Android models, Android OS versions, and\n          Android locales which are available as test targets. Also run tests\n          against your Android app on these devices, monitor your test progress,\n          and view detailed test results in the Firebase console.\n          ', 'EXAMPLES': '          To see a list of available Android devices, their form factors, and\n          supported Android OS versions, run:\n\n            $ {command} models list\n\n          To view more detailed information about a specific Android model, run:\n\n            $ {command} models describe MODEL_ID\n\n          To view details about available Android OS versions, such as their\n          code names and release dates, run:\n\n            $ {command} versions list\n\n          To view information about a specific Android OS version, run:\n\n            $ {command} versions describe VERSION_ID\n\n          To view the list of available Android locales which can be used for\n          testing internationalized applications, run:\n\n            $ {command} locales list\n\n          To view information about a specific locale, run:\n\n            $ {command} locales describe LOCALE\n\n          To view all options available for running Android tests, run:\n\n            $ {command} run --help\n      '}

    @staticmethod
    def Args(parser):
        """Method called by Calliope to register flags common to this sub-group.

    Args:
      parser: An argparse parser used to add arguments that immediately follow
          this group in the CLI. Positional arguments are allowed.
    """