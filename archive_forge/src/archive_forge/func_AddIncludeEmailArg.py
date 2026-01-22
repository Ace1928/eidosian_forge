from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddIncludeEmailArg(parser):
    parser.add_argument('--include-email', action='store_true', help="Specify whether or not service account email is included in the identity token. If specified, the token will contain 'email' and 'email_verified' claims. This flag should only be used for impersonate service account.")