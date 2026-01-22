from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddPlatformVersion(parser, required=True):
    """Adds platform version flag.

  Args:
    parser: The argparse.parser to add the arguments to.
    required: Indicates if the flag is required.
  """
    help_text = "\nPlatform version to use for the cluster.\n\nTo retrieve a list of valid versions, run:\n\n  $ gcloud alpha container attached get-server-config --location=LOCATION\n\nReplace ``LOCATION'' with the target Google Cloud location for the cluster.\n"
    parser.add_argument('--platform-version', required=required, help=help_text)