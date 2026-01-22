from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddAccountArg(parser):
    parser.add_argument('account', nargs='?', help='Account to print the identity token for. If not specified, the current active account will be used.')