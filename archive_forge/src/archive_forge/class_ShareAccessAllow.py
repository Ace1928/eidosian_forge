import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class ShareAccessAllow(command.ShowOne):
    """Create a new share access rule."""
    _description = _('Create new share access rule')

    def get_parser(self, prog_name):
        parser = super(ShareAccessAllow, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of the NAS share to modify.'))
        parser.add_argument('access_type', metavar='<access_type>', help=_('Access rule type (only "ip", "user" (user or group), "cert" or "cephx" are supported).'))
        parser.add_argument('access_to', metavar='<access_to>', help=_('Value that defines access.'))
        parser.add_argument('--properties', type=str, nargs='*', metavar='<key=value>', help=_('Space separated list of key=value pairs of properties. OPTIONAL: Default=None. Available only for API microversion >= 2.45.'))
        parser.add_argument('--access-level', metavar='<access_level>', type=str, default=None, choices=['rw', 'ro'], help=_('Share access level ("rw" and "ro" access levels are supported). Defaults to rw.'))
        parser.add_argument('--wait', action='store_true', help=_('Wait for share access rule creation.'))
        parser.add_argument('--lock-visibility', action='store_true', default=False, help=_('Whether the sensitive fields of the access rule redacted to other users. Only available with API version >= 2.82.'))
        parser.add_argument('--lock-deletion', action='store_true', default=False, help=_("When enabled, a 'delete' lock will be placed against the rule and the rule cannot be deleted while the lock exists. Only available with API version >= 2.82."))
        parser.add_argument('--lock-reason', metavar='<lock_reason>', type=str, default=None, help=_('Reason for locking the access rule. Should only be provided alongside a deletion or visibility lock. Only available with API version >= 2.82.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = apiutils.find_resource(share_client.shares, parsed_args.share)
        lock_kwargs = {}
        if parsed_args.lock_visibility:
            lock_kwargs['lock_visibility'] = parsed_args.lock_visibility
        if parsed_args.lock_deletion:
            lock_kwargs['lock_deletion'] = parsed_args.lock_deletion
        if parsed_args.lock_reason:
            lock_kwargs['lock_reason'] = parsed_args.lock_reason
        if lock_kwargs and share_client.api_version < api_versions.APIVersion('2.82'):
            raise exceptions.CommandError('Restricted access rules are only available starting from API version 2.82.')
        if lock_kwargs.get('lock_reason', None) and (not (lock_kwargs.get('lock_visibility', None) or lock_kwargs.get('lock_deletion', None))):
            raise exceptions.CommandError('Lock reason can only be set while locking the deletion or visibility.')
        properties = {}
        if parsed_args.properties:
            if share_client.api_version >= api_versions.APIVersion('2.45'):
                properties = utils.extract_properties(parsed_args.properties)
            else:
                raise exceptions.CommandError('Adding properties to access rules is supported only with API microversion 2.45 and beyond')
        try:
            share_access_rule = share.allow(access_type=parsed_args.access_type, access=parsed_args.access_to, access_level=parsed_args.access_level, metadata=properties, **lock_kwargs)
            if parsed_args.wait:
                if not oscutils.wait_for_status(status_f=share_client.share_access_rules.get, res_id=share_access_rule['id'], status_field='state'):
                    LOG.error(_('ERROR: Share access rule is in error state.'))
                share_access_rule = oscutils.find_resource(share_client.share_access_rules, share_access_rule['id'])._info
            share_access_rule.update({'properties': utils.format_properties(share_access_rule.pop('metadata', {}))})
            return (ACCESS_RULE_ATTRIBUTES, oscutils.get_dict_properties(share_access_rule, ACCESS_RULE_ATTRIBUTES))
        except Exception as e:
            raise exceptions.CommandError("Failed to create access to share '%s': %s" % (share, e))