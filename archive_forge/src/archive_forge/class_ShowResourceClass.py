from osc_lib.command import command
from osc_lib import utils
from osc_placement import version
class ShowResourceClass(command.ShowOne):
    """Return a representation of the resource class identified by ``<name>``.

    This command requires at least ``--os-placement-api-version 1.2``.
    """

    def get_parser(self, prog_name):
        parser = super(ShowResourceClass, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help='Name of the resource class')
        return parser

    @version.check(version.ge('1.2'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = PER_CLASS_URL.format(name=parsed_args.name)
        resource = http.request('GET', url).json()
        return (FIELDS, utils.get_dict_properties(resource, FIELDS))