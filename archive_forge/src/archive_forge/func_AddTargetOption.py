from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddTargetOption(parser):
    parser.add_argument('--target', metavar='(ID|DESCRIPTION_REGEXP)', help="          The debug target. It may be a target ID or name obtained from\n          'debug targets list', or it may be a regular expression uniquely\n          specifying a debuggee based on its description or name. For App\n          Engine projects, if not specified, the default target is\n          the most recent deployment of the default module and version.\n      ")