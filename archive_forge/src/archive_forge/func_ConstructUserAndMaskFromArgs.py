from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
def ConstructUserAndMaskFromArgs(alloydb_messages, user_ref, args):
    """Validates command line arguments and creates the user and field mask.

  Args:
    alloydb_messages: Messages module for the API client.
    user_ref: resource path of the resource being updated
    args: Command line input arguments.

  Returns:
    An AlloyDB user and mask for update.
  """
    password_path = 'password'
    database_roles_path = 'database_roles'
    user_resource = alloydb_messages.User()
    user_resource.name = user_ref.RelativeName()
    if 'set-password' in args.command_path:
        user_resource.password = args.password
        return (user_resource, password_path)
    if 'set-roles' in args.command_path:
        user_resource.databaseRoles = args.db_roles
        return (user_resource, database_roles_path)
    if 'set-superuser' in args.command_path:
        if args.superuser:
            args.db_roles.append('alloydbsuperuser')
        else:
            args.db_roles.remove('alloydbsuperuser')
        user_resource.databaseRoles = args.db_roles
        return (user_resource, database_roles_path)
    return (user_resource, None)