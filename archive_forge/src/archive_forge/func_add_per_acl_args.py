from cliff import command
from cliff import lister
from barbicanclient.v1 import acls
def add_per_acl_args(self, parser):
    parser.add_argument('--user', '-u', action='append', default=None, nargs='?', dest='users', help='Keystone userid(s) for ACL.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--project-access', dest='project_access', action='store_true', default=None, help='Flag to enable project access behavior.')
    group.add_argument('--no-project-access', dest='project_access', action='store_false', help='Flag to disable project access behavior.')
    parser.add_argument('--operation-type', '-o', default=acls.DEFAULT_OPERATION_TYPE, dest='operation_type', choices=['read'], help='Type of Barbican operation ACL is set for')