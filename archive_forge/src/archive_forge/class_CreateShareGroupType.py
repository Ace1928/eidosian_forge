import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class CreateShareGroupType(command.ShowOne):
    """Create new share group type."""
    _description = _('Create new share group type')
    log = logging.getLogger(__name__ + '.CreateShareGroupType')

    def get_parser(self, prog_name):
        parser = super(CreateShareGroupType, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', default=None, help=_('Share group type name'))
        parser.add_argument('share_types', metavar='<share-types>', nargs='+', default=None, help=_('List of share type names or IDs. Example: my-share-type-1 my-share-type-2'))
        parser.add_argument('--group-specs', type=str, nargs='*', metavar='<key=value>', default=None, help=_('Share Group type extra specs by key and value. OPTIONAL: Default=None. Example: --group-specs consistent_snapshot_support=host.'))
        parser.add_argument('--public', metavar='<public>', default=True, help=_('Make type accessible to the public (default true).'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        kwargs = {'name': parsed_args.name}
        share_types_list = []
        for share_type in parsed_args.share_types:
            try:
                share_type_obj = apiutils.find_resource(share_client.share_types, share_type)
                share_types_list.append(share_type_obj.name)
            except Exception as e:
                msg = LOG.error(_("Failed to find the share type with name or ID '%(share_type)s': %(e)s"), {'share_type': share_type, 'e': e})
                raise exceptions.CommandError(msg)
        kwargs['share_types'] = share_types_list
        if parsed_args.public:
            kwargs['is_public'] = strutils.bool_from_string(parsed_args.public, default=True)
        group_specs = {}
        if parsed_args.group_specs:
            for item in parsed_args.group_specs:
                group_specs = utils.extract_group_specs(group_specs, [item])
        kwargs['group_specs'] = group_specs
        share_group_type = share_client.share_group_types.create(**kwargs)
        formatter = parsed_args.formatter
        formatted_group_type = utils.format_share_group_type(share_group_type, formatter)
        return (ATTRIBUTES, oscutils.get_dict_properties(formatted_group_type, ATTRIBUTES))