from cliff import lister
from cliff import show
from barbicanclient.v1 import cas
class ListCA(lister.Lister):
    """List CAs."""

    def get_parser(self, prog_name):
        parser = super(ListCA, self).get_parser(prog_name)
        parser.add_argument('--limit', '-l', default=10, help='specify the limit to the number of items to list per page (default: %(default)s; maximum: 100)', type=int)
        parser.add_argument('--offset', '-o', default=0, help='specify the page offset (default: %(default)s)', type=int)
        parser.add_argument('--name', '-n', default=None, help='specify the ca name (default: %(default)s)')
        return parser

    def take_action(self, args):
        obj_list = self.app.client_manager.key_manager.cas.list(args.limit, args.offset, args.name)
        return cas.CA._list_objects(obj_list)