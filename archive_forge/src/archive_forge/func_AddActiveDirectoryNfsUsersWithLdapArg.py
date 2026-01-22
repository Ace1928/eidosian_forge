from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddActiveDirectoryNfsUsersWithLdapArg(parser):
    """Adds a --nfs-users-with-ldap arg to the given parser."""
    parser.add_argument('--nfs-users-with-ldap', type=arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), help='Boolean flag that allows access to local users and LDAP users. If access is needed only for LDAP users, it has to be disabled.')