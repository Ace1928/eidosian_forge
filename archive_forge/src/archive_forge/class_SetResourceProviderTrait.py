from osc_lib.command import command
from osc_placement import version
class SetResourceProviderTrait(command.Lister):
    """Associate traits with the resource provider identified by {uuid}.

    All the associated traits will be replaced by the traits specified.

    This command requires at least ``--os-placement-api-version 1.6``.
    """

    def get_parser(self, prog_name):
        parser = super(SetResourceProviderTrait, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider.')
        parser.add_argument('--trait', metavar='<trait>', help='Name of the trait. May be repeated.', default=[], action='append')
        return parser

    @version.check(version.ge('1.6'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = RP_BASE_URL.format(uuid=parsed_args.uuid)
        rp = http.request('GET', url).json()
        url = RP_TRAITS_URL.format(uuid=parsed_args.uuid)
        payload = {'resource_provider_generation': rp['generation'], 'traits': parsed_args.trait}
        traits = http.request('PUT', url, json=payload).json()['traits']
        return (FIELDS, [[t] for t in traits])