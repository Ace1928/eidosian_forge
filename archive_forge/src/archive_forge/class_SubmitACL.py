from cliff import command
from cliff import lister
from barbicanclient.v1 import acls
class SubmitACL(lister.Lister, ArgMixin):
    """Submit ACL on a secret or container as identified by its href."""

    def get_parser(self, prog_name):
        parser = super(SubmitACL, self).get_parser(prog_name)
        self.add_ref_arg(parser)
        self.add_per_acl_args(parser)
        return parser

    def take_action(self, args):
        """Submit complete secret or container ACL settings to Barbican

        This action replaces existing ACL setting on server with provided
        inputs.

        :returns: List of objects for valid entity_ref
        :rtype: :class:`barbicanclient.acls.SecretACL` or
            :class:`barbicanclient.acls.ContainerACL`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        entity = self.create_acl_entity_from_args(self.app.client_manager.key_manager.acls, args)
        entity.submit()
        entity.load_acls_data()
        return self.get_acls_as_lister(entity)