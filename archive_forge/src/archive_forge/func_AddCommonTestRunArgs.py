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
def AddCommonTestRunArgs(parser):
    """Register args which are common to all 'gcloud test run' commands.

  Args:
    parser: An argparse parser used to add arguments that follow a command
        in the CLI.
  """
    parser.add_argument('argspec', nargs='?', completer=arg_file.ArgSpecCompleter, help='An ARG_FILE:ARG_GROUP_NAME pair, where ARG_FILE is the path to a file containing groups of test arguments in yaml format, and ARG_GROUP_NAME is the particular yaml object holding a group of arg:value pairs to use. Run *$ gcloud topic arg-files* for more information and examples.')
    parser.add_argument('--async', action='store_true', default=None, dest='async_', help='Invoke a test asynchronously without waiting for test results.')
    parser.add_argument('--client-details', type=arg_parsers.ArgDict(), metavar='KEY=VALUE', help='      Comma-separated, KEY=VALUE map of additional details to attach to the test\n      matrix. Arbitrary KEY=VALUE pairs may be attached to a test matrix to\n      provide additional context about the tests being run. When consuming the\n      test results, such as in Cloud Functions or a CI system, these details can\n      add additional context such as a link to the corresponding pull request.\n\n      Example:\n\n      ```\n      --client-details=buildNumber=1234,pullRequest=https://example.com/link/to/pull-request\n      ```\n\n      To help you identify and locate your test matrix in the Firebase console,\n      use the matrixLabel key.\n\n      Example:\n\n      ```\n      --client-details=matrixLabel="Example matrix label"\n      ```\n      ')
    parser.add_argument('--num-flaky-test-attempts', metavar='int', type=arg_validate.NONNEGATIVE_INT_PARSER, help='      Specifies the number of times a test execution should be reattempted if\n      one or more of its test cases fail for any reason. An execution that\n      initially fails but succeeds on any reattempt is reported as FLAKY.\n\n      The maximum number of reruns allowed is 10. (Default: 0, which implies\n      no reruns.) All additional attempts are executed in parallel.\n      ')
    parser.add_argument('--record-video', action='store_true', default=None, help='Enable video recording during the test. Enabled by default, use --no-record-video to disable.')
    parser.add_argument('--results-bucket', help='The name of a Google Cloud Storage bucket where raw test results will be stored (default: "test-lab-<random-UUID>"). Note that the bucket must be owned by a billing-enabled project, and that using a non-default bucket will result in billing charges for the storage used.')
    parser.add_argument('--results-dir', help='The name of a *unique* Google Cloud Storage object within the results bucket where raw test results will be stored (default: a timestamp with a random suffix). Caution: if specified, this argument *must be unique* for each test matrix you create, otherwise results from multiple test matrices will be overwritten or intermingled.')
    parser.add_argument('--timeout', category=base.COMMONLY_USED_FLAGS, type=arg_validate.TIMEOUT_PARSER, help='The max time this test execution can run before it is cancelled (default: 15m). It does not include any time necessary to prepare and clean up the target device. The maximum possible testing time is 45m on physical devices and 60m on virtual devices. The _TIMEOUT_ units can be h, m, or s. If no unit is given, seconds are assumed. Examples:\n- *--timeout 1h* is 1 hour\n- *--timeout 5m* is 5 minutes\n- *--timeout 200s* is 200 seconds\n- *--timeout 100* is 100 seconds')