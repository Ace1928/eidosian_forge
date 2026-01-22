from cliff import command
from cliff import lister
from barbicanclient.v1 import acls
class GetACLs(lister.Lister, ArgMixin):
    """Retrieve ACLs for a secret or container by providing its href."""

    def get_parser(self, prog_name):
        parser = super(GetACLs, self).get_parser(prog_name)
        self.add_ref_arg(parser)
        return parser

    def take_action(self, args):
        """Retrieves a secret or container ACL settings from Barbican.

        This action provides list of all ACL settings for a secret or container
        in Barbican.

        :returns: List of objects for valid entity_ref
        :rtype: :class:`barbicanclient.acls.SecretACL` or
            :class:`barbicanclient.acls.ContainerACL`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        blank_entity = self.create_blank_acl_entity_from_uri(self.app.client_manager.key_manager.acls, args)
        acl_entity = self.app.client_manager.key_manager.acls.get(blank_entity.entity_ref)
        return self.get_acls_as_lister(acl_entity)