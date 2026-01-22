from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
class ReplaceTag(neutronv20.NeutronCommand):
    """Replace all tags on the resource."""

    def get_parser(self, prog_name):
        parser = super(ReplaceTag, self).get_parser(prog_name)
        _add_common_arguments(parser)
        parser.add_argument('--tag', metavar='TAG', action='append', dest='tags', required=True, help=_('Tag (This option can be repeated).'))
        return parser

    def take_action(self, parsed_args):
        client = self.get_client()
        resource_type, resource_id = _convert_resource_args(client, parsed_args)
        body = {'tags': parsed_args.tags}
        client.replace_tag(resource_type, resource_id, body)