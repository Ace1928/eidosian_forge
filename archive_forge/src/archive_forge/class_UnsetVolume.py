import argparse
import copy
import functools
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class UnsetVolume(command.Command):
    _description = _('Unset volume properties')

    def get_parser(self, prog_name):
        parser = super(UnsetVolume, self).get_parser(prog_name)
        parser.add_argument('volume', metavar='<volume>', help=_('Volume to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key>', action='append', help=_('Remove a property from volume (repeat option to remove multiple properties)'))
        parser.add_argument('--image-property', metavar='<key>', action='append', help=_('Remove an image property from volume (repeat option to remove multiple image properties)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        volume = utils.find_resource(volume_client.volumes, parsed_args.volume)
        result = 0
        if parsed_args.property:
            try:
                volume_client.volumes.delete_metadata(volume.id, parsed_args.property)
            except Exception as e:
                LOG.error(_('Failed to unset volume property: %s'), e)
                result += 1
        if parsed_args.image_property:
            try:
                volume_client.volumes.delete_image_metadata(volume.id, parsed_args.image_property)
            except Exception as e:
                LOG.error(_('Failed to unset image property: %s'), e)
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('One or more of the unset operations failed'))