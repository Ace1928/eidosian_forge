import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from heatclient.common import format_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class ResourceTypeShow(format_utils.YamlFormat):
    """Show details and optionally generate a template for a resource type."""
    log = logging.getLogger(__name__ + '.ResourceTypeShow')

    def get_parser(self, prog_name):
        parser = super(ResourceTypeShow, self).get_parser(prog_name)
        parser.add_argument('resource_type', metavar='<resource-type>', help=_('Resource type to show details for'))
        parser.add_argument('--template-type', metavar='<template-type>', help=_('Optional template type to generate, hot or cfn'))
        parser.add_argument('--long', default=False, action='store_true', help=_('Show resource type with corresponding description.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        if parsed_args.template_type is not None and parsed_args.long:
            msg = _('Cannot use --template-type and --long in one time.')
            raise exc.CommandError(msg)
        heat_client = self.app.client_manager.orchestration
        return _show_resourcetype(heat_client, parsed_args)