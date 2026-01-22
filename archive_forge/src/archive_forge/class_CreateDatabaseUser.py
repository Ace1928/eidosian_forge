from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
class CreateDatabaseUser(command.Command):
    _description = _('Creates a user on an instance.')

    def get_parser(self, prog_name):
        parser = super(CreateDatabaseUser, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        parser.add_argument('name', metavar='<name>', help=_('Name of user.'))
        parser.add_argument('password', metavar='<password>', help=_('Password of user.'))
        parser.add_argument('--host', metavar='<host>', help=_('Optional host of user.'))
        parser.add_argument('--databases', metavar='<databases>', nargs='+', default=[], help=_('Optional list of databases.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        users = manager.users
        instance = utils.find_resource(manager.instances, parsed_args.instance)
        databases = [{'name': value} for value in parsed_args.databases]
        user = {'name': parsed_args.name, 'password': parsed_args.password, 'databases': databases}
        if parsed_args.host:
            user['host'] = parsed_args.host
        users.create(instance, [user])