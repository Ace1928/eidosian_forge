import os
from cliff import command
from cliff import lister
from cliff import show
from barbicanclient.v1 import secrets
class UpdateSecret(command.Command):
    """Update a secret with no payload in Barbican."""

    def get_parser(self, prog_name):
        parser = super(UpdateSecret, self).get_parser(prog_name)
        parser.add_argument('URI', help='The URI reference for the secret.')
        parser.add_argument('payload', help='the unencrypted secret')
        return parser

    def take_action(self, args):
        self.app.client_manager.key_manager.secrets.update(args.URI, args.payload)