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
def AddMatrixArgs(parser):
    """Register the repeatable args which define the axes for a test matrix.

  Args:
    parser: An argparse parser used to add arguments that follow a command
        in the CLI.
  """
    parser.add_argument('--device', category=base.COMMONLY_USED_FLAGS, type=arg_parsers.ArgDict(min_length=1), action='append', metavar='DIMENSION=VALUE', help="      A list of ``DIMENSION=VALUE'' pairs which specify a target device to test\n      against. This flag may be repeated to specify multiple devices. The four\n      device dimensions are: *model*, *version*, *locale*, and *orientation*. If\n      any dimensions are omitted, they will use a default value. The default\n      value, and all possible values, for each dimension can be found with the\n      ``list'' command for that dimension, such as `$ {parent_command} models\n      list`. *--device* is now the preferred way to specify test devices and may\n      not be used in conjunction with *--devices-ids*, *--os-version-ids*,\n      *--locales*, or *--orientations*. Omitting all of the preceding\n      dimension-related flags will run tests against a single device using\n      defaults for all four device dimensions.\n\n      Examples:\n\n      ```\n      --device model=Nexus6\n      --device version=23,orientation=portrait\n      --device model=shamu,version=22,locale=zh_CN,orientation=default\n      ```\n      ")
    parser.add_argument('--device-ids', '-d', category=DEPRECATED_DEVICE_DIMENSIONS, type=arg_parsers.ArgList(min_length=1), metavar='MODEL_ID', help='The list of MODEL_IDs to test against (default: one device model determined by the Firebase Test Lab device catalog; see TAGS listed by the `$ {parent_command} models list` command).')
    parser.add_argument('--os-version-ids', '-v', category=DEPRECATED_DEVICE_DIMENSIONS, type=arg_parsers.ArgList(min_length=1), metavar='OS_VERSION_ID', help='The list of OS_VERSION_IDs to test against (default: a version ID determined by the Firebase Test Lab device catalog).')
    parser.add_argument('--locales', '-l', category=DEPRECATED_DEVICE_DIMENSIONS, type=arg_parsers.ArgList(min_length=1), metavar='LOCALE', help='The list of LOCALEs to test against (default: a single locale determined by the Firebase Test Lab device catalog).')
    parser.add_argument('--orientations', '-o', category=DEPRECATED_DEVICE_DIMENSIONS, type=arg_parsers.ArgList(min_length=1, max_length=2, choices=arg_validate.ORIENTATION_LIST), completer=arg_parsers.GetMultiCompleter(OrientationsCompleter), metavar='ORIENTATION', help="The device orientation(s) to test against (default: portrait). Specifying 'default' will pick the preferred orientation for the app.")