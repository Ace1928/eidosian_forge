from cliff import command
from cliff import lister
from barbicanclient.v1 import acls
class ArgMixin(object):
    """Mixin class for CLI arguments and validation"""

    def add_ref_arg(self, parser):
        parser.add_argument('URI', help='The URI reference for the secret or container.')

    def add_per_acl_args(self, parser):
        parser.add_argument('--user', '-u', action='append', default=None, nargs='?', dest='users', help='Keystone userid(s) for ACL.')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--project-access', dest='project_access', action='store_true', default=None, help='Flag to enable project access behavior.')
        group.add_argument('--no-project-access', dest='project_access', action='store_false', help='Flag to disable project access behavior.')
        parser.add_argument('--operation-type', '-o', default=acls.DEFAULT_OPERATION_TYPE, dest='operation_type', choices=['read'], help='Type of Barbican operation ACL is set for')

    def create_blank_acl_entity_from_uri(self, acl_manager, args):
        """Validates URI argument and creates blank ACL entity"""
        entity = acl_manager.create(args.URI)
        entity.validate_input_ref()
        return entity

    def create_acl_entity_from_args(self, acl_manager, args):
        blank_entity = self.create_blank_acl_entity_from_uri(acl_manager, args)
        users = args.users
        if users is None:
            users = []
        else:
            users = [user for user in users if user is not None]
        entity = acl_manager.create(entity_ref=blank_entity.entity_ref, users=users, project_access=args.project_access, operation_type=args.operation_type)
        return entity

    def get_acls_as_lister(self, acl_entity):
        """Gets per operation ACL data in expected format for lister command"""
        for acl in acl_entity.operation_acls:
            setattr(acl, 'columns', acl_entity.columns)
        return acls.ACLFormatter._list_objects(acl_entity.operation_acls)