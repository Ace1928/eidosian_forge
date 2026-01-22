from cliff import command
from cliff import lister
from cliff import show
from barbicanclient.v1.containers import CertificateContainer
from barbicanclient.v1.containers import Container
from barbicanclient.v1.containers import RSAContainer
class GetContainer(show.ShowOne):
    """Retrieve a container by providing its URI."""

    def get_parser(self, prog_name):
        parser = super(GetContainer, self).get_parser(prog_name)
        parser.add_argument('URI', help='The URI reference for the container.')
        return parser

    def take_action(self, args):
        entity = self.app.client_manager.key_manager.containers.get(args.URI)
        return entity._get_formatted_entity()