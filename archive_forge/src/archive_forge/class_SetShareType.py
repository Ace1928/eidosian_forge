import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
from manilaclient.osc import utils
class SetShareType(command.Command):
    """Set share type properties."""
    _description = _('Set share type properties')

    def get_parser(self, prog_name):
        parser = super(SetShareType, self).get_parser(prog_name)
        parser.add_argument('share_type', metavar='<share_type>', help=_('Name or ID of the share type to modify'))
        parser.add_argument('--extra-specs', type=str, nargs='*', metavar='<key=value>', default=None, help=_("Extra specs key and value of share type that will be used for share type creation. OPTIONAL: Default=None. example --extra-specs  thin_provisioning='<is> True', replication_type=readable."))
        parser.add_argument('--public', metavar='<public>', default=None, help=_('New visibility of the share type. If set to True, share type will be available to all projects in the cloud. Available only for microversion >= 2.50'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('New description of share type. Available only for microversion >= 2.50'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('New name of share type. Available only for microversion >= 2.50'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_type = apiutils.find_resource(share_client.share_types, parsed_args.share_type)
        can_update = share_client.api_version >= api_versions.APIVersion('2.50')
        kwargs = {}
        if parsed_args.name is not None:
            if can_update:
                kwargs['name'] = parsed_args.name
            else:
                raise exceptions.CommandError('Setting (new) name to share type is only available with API microversion >= 2.50')
        if parsed_args.description is not None:
            if can_update:
                kwargs['description'] = parsed_args.description
            else:
                raise exceptions.CommandError('Setting (new) description to share type is only available with API microversion >= 2.50')
        if parsed_args.public is not None:
            if can_update:
                kwargs['is_public'] = strutils.bool_from_string(parsed_args.public, default=True)
            else:
                raise exceptions.CommandError('Setting visibility to share type is only available with API microversion >= 2.50')
        if kwargs:
            share_type.update(**kwargs)
        if parsed_args.extra_specs:
            extra_specs = utils.extract_extra_specs(extra_specs={}, specs_to_add=parsed_args.extra_specs)
            try:
                share_type.set_keys(extra_specs)
            except Exception as e:
                raise exceptions.CommandError('Failed to set share type key: %s' % e)