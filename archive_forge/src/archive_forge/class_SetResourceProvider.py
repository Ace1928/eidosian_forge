import argparse
from osc_lib.command import command
from osc_lib import utils
from osc_placement.resources import common
from osc_placement import version
class SetResourceProvider(command.ShowOne, version.CheckerMixin):
    """Update an existing resource provider"""

    def get_parser(self, prog_name):
        parser = super(SetResourceProvider, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider')
        parser.add_argument('--name', metavar='<name>', help='A new name of the resource provider', required=True)
        parser.add_argument('--parent-provider', metavar='<parent_provider>', help='UUID of the parent provider. Can only be set if the resource provider has no parent yet. This option requires at least ``--os-placement-api-version 1.14``.')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL + '/' + parsed_args.uuid
        data = dict(name=parsed_args.name)
        if 'parent_provider' in parsed_args and parsed_args.parent_provider:
            self.check_version(version.ge('1.14'))
            data['parent_provider_uuid'] = parsed_args.parent_provider
        resource = http.request('PUT', url, json=data).json()
        fields = ('uuid', 'name', 'generation')
        if self.compare_version(version.ge('1.14')):
            fields += ('root_provider_uuid', 'parent_provider_uuid')
        return (fields, utils.get_dict_properties(resource, fields))