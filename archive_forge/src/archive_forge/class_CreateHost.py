import logging
from blazarclient import command
from blazarclient import exception
class CreateHost(command.CreateCommand):
    """Create a host."""
    resource = 'host'
    json_indent = 4
    log = logging.getLogger(__name__ + '.CreateHost')

    def get_parser(self, prog_name):
        parser = super(CreateHost, self).get_parser(prog_name)
        parser.add_argument('name', metavar=self.resource.upper(), help='Name of the host to add')
        parser.add_argument('--extra', metavar='<key>=<value>', action='append', dest='extra_capabilities', default=[], help='Extra capabilities key/value pairs to add for the host')
        return parser

    def args2body(self, parsed_args):
        params = {}
        if parsed_args.name:
            params['name'] = parsed_args.name
        extras = {}
        if parsed_args.extra_capabilities:
            for capa in parsed_args.extra_capabilities:
                key, _sep, value = capa.partition('=')
                extras[key] = value
            params.update(extras)
        return params