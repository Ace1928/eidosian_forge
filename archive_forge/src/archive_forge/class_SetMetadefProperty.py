import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetMetadefProperty(command.Command):
    _description = _('Update metadef namespace property')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', help=_('Internal name of the property'))
        parser.add_argument('--title', help=_('Property name displayed to the user'))
        parser.add_argument('--type', help=_('Property type'))
        parser.add_argument('--schema', help=_('Valid JSON schema of the property'))
        parser.add_argument('namespace', help=_('Namespace of the namespace to which the property belongs'))
        parser.add_argument('property', help=_('Property to update'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        data = image_client.get_metadef_property(parsed_args.property, parsed_args.namespace)
        kwargs = _format_property(data)
        for key in ['name', 'title', 'type']:
            argument = getattr(parsed_args, key, None)
            if argument is not None:
                kwargs[key] = argument
        if parsed_args.schema:
            try:
                kwargs.update(json.loads(parsed_args.schema))
            except json.JSONDecodeError as e:
                raise exceptions.CommandError(_('Failed to load JSON schema: %(e)s') % {'e': e})
        image_client.update_metadef_property(parsed_args.property, parsed_args.namespace, **kwargs)