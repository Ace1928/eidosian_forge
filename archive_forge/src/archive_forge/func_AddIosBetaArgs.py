from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
import six
def AddIosBetaArgs(parser):
    """Register args which are only available in the iOS beta run command.

  Args:
    parser: An argparse parser used to add args that follow a command.
  """
    parser.add_argument('--additional-ipas', type=arg_parsers.ArgList(min_length=1, max_length=100), metavar='IPA', help='List of up to 100 additional IPAs to install, in addition to the one being directly tested. The path may be in the local filesystem or in Google Cloud Storage using gs:// notation.')
    parser.add_argument('--other-files', type=arg_parsers.ArgDict(min_length=1), metavar='DEVICE_PATH=FILE_PATH', help='      A list of device-path=file-path pairs that specify the paths of the test\n      device and the files you want pushed to the device prior to testing.\n\n      Device paths should either be under the Media shared folder (e.g. prefixed\n      with /private/var/mobile/Media) or within the documents directory of the\n      filesystem of an app under test (e.g. /Documents). Device paths to app\n      filesystems should be prefixed by the bundle ID and a colon. Source file\n      paths may be in the local filesystem or in Google Cloud Storage\n      (gs://...).\n\n      Examples:\n\n      ```\n      --other-files com.my.app:/Documents/file.txt=local/file.txt,/private/var/mobile/Media/file.jpg=gs://bucket/file.jpg\n      ```\n      ')
    parser.add_argument('--directories-to-pull', type=arg_parsers.ArgList(), metavar='DIR_TO_PULL', help="      A list of paths that will be copied from the device's storage to\n      the designated results bucket after the test is complete. These must be\n      absolute paths under `/private/var/mobile/Media` or `/Documents` of the\n      app under test. If the path is under an app's `/Documents`, it must be\n      prefixed with the app's bundle id and a colon.\n\n      Example:\n\n      ```\n      --directories-to-pull=com.my.app:/Documents/output,/private/var/mobile/Media/output\n      ```\n      ")
    parser.add_argument('--scenario-numbers', metavar='int', type=arg_parsers.ArgList(element_type=int, min_length=1, max_length=1024), help='A list of game-loop scenario numbers which will be run as part of the test (default: scenario 1). A maximum of 1024 scenarios may be specified in one test matrix, but the maximum number may also be limited by the overall test *--timeout* setting. This flag is only valid when *--type=game-loop* is also set.')
    parser.add_argument('--robo-script', help='      The path to a Robo Script JSON file. The path may be in the local\n      filesystem or in Google Cloud Storage using gs:// notation. You can\n      guide the Robo test to perform specific actions by specifying a Robo\n      Script with this argument. Learn more at\n      https://firebase.google.com/docs/test-lab/robo-ux-test#scripting.\n      This flag is only valid when *--type=robo* is also set.\n      ')