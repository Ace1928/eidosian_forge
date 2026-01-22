import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CacheImageForAggregate(command.Command):
    _description = _('Request image caching for aggregate')

    def get_parser(self, prog_name):
        parser = super(CacheImageForAggregate, self).get_parser(prog_name)
        parser.add_argument('aggregate', metavar='<aggregate>', help=_('Aggregate (name or ID)'))
        parser.add_argument('image', metavar='<image>', nargs='+', help=_('Image ID to request caching for aggregate (name or ID). May be specified multiple times.'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        if not sdk_utils.supports_microversion(compute_client, '2.81'):
            msg = _('This operation requires server support for API microversion 2.81')
            raise exceptions.CommandError(msg)
        aggregate = compute_client.find_aggregate(parsed_args.aggregate, ignore_missing=False)
        images = []
        for img in parsed_args.image:
            image = self.app.client_manager.sdk_connection.image.find_image(img, ignore_missing=False)
            images.append(image.id)
        compute_client.aggregate_precache_images(aggregate.id, images)