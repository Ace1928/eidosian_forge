from osc_lib.command import command
from osc_placement import version
class ListTrait(command.Lister):
    """Return a list of valid trait strings.

    This command requires at least ``--os-placement-api-version 1.6``.
    """

    def get_parser(self, prog_name):
        parser = super(ListTrait, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help='A string to filter traits. The following options are available: startswith operator filters the traits whose name begins with a specific prefix, e.g. name=startswith:CUSTOM, in operator filters the traits whose name is in the specified list, e.g. name=in:HW_CPU_X86_AVX,HW_CPU_X86_SSE, HW_CPU_X86_INVALID_FEATURE.')
        parser.add_argument('--associated', action='store_true', help='If this parameter is presented, the returned traits will be those that are associated with at least one resource provider.')
        return parser

    @version.check(version.ge('1.6'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL
        params = {}
        if parsed_args.name:
            params['name'] = parsed_args.name
        if parsed_args.associated:
            params['associated'] = parsed_args.associated
        traits = http.request('GET', url, params=params).json()['traits']
        return (FIELDS, [[t] for t in traits])