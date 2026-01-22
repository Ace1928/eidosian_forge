import argparse
from base64 import b64encode
import logging
import os
import sys
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack.image import image_signer
from osc_lib.api import utils as api_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.common import progressbar
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ImportImage(command.ShowOne):
    _description = _('Initiate the image import process.\nThis requires support for the interoperable image import process, which was first introduced in Image API version 2.6 (Glance 16.0.0 (Queens))')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('image', metavar='<image>', help=_('Image to initiate import process for (name or ID)'))
        parser.add_argument('--method', metavar='<method>', default='glance-direct', dest='import_method', choices=['glance-direct', 'web-download', 'glance-download', 'copy-image'], help=_("Import method used for image import process. Not all deployments will support all methods. The 'glance-direct' method (default) requires images be first staged using the 'image-stage' command."))
        parser.add_argument('--uri', metavar='<uri>', help=_("URI to download the external image (only valid with the 'web-download' import method)"))
        parser.add_argument('--remote-image', metavar='<REMOTE_IMAGE>', help=_("The image of remote glance (ID only) to be imported (only valid with the 'glance-download' import method)"))
        parser.add_argument('--remote-region', metavar='<REMOTE_GLANCE_REGION>', help=_("The remote Glance region to download the image from (only valid with the 'glance-download' import method)"))
        parser.add_argument('--remote-service-interface', metavar='<REMOTE_SERVICE_INTERFACE>', help=_("The remote Glance service interface to use when importing images (only valid with the 'glance-download' import method)"))
        stores_group = parser.add_mutually_exclusive_group()
        stores_group.add_argument('--store', metavar='<STORE>', dest='stores', nargs='*', help=_("Backend store to upload image to (specify multiple times to upload to multiple stores) (either '--store' or '--all-stores' required with the 'copy-image' import method)"))
        stores_group.add_argument('--all-stores', help=_("Make image available to all stores (either '--store' or '--all-stores' required with the 'copy-image' import method)"))
        parser.add_argument('--allow-failure', action='store_true', dest='allow_failure', default=True, help=_('When uploading to multiple stores, indicate that the import should be continue should any of the uploads fail. Only usable with --stores or --all-stores'))
        parser.add_argument('--disallow-failure', action='store_true', dest='allow_failure', default=True, help=_('When uploading to multiple stores, indicate that the import should be reverted should any of the uploads fail. Only usable with --stores or --all-stores'))
        parser.add_argument('--wait', action='store_true', help=_('Wait for operation to complete'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        try:
            import_info = image_client.get_import_info()
        except sdk_exceptions.ResourceNotFound:
            msg = _('The Image Import feature is not supported by this deployment')
            raise exceptions.CommandError(msg)
        import_methods = import_info.import_methods['value']
        if parsed_args.import_method not in import_methods:
            msg = _("The '%s' import method is not supported by this deployment. Supported: %s")
            raise exceptions.CommandError(msg % (parsed_args.import_method, ', '.join(import_methods)))
        if parsed_args.import_method == 'web-download':
            if not parsed_args.uri:
                msg = _("The '--uri' option is required when using '--method=web-download'")
                raise exceptions.CommandError(msg)
        elif parsed_args.uri:
            msg = _("The '--uri' option is only supported when using '--method=web-download'")
            raise exceptions.CommandError(msg)
        if parsed_args.import_method == 'glance-download':
            if not (parsed_args.remote_region and parsed_args.remote_image):
                msg = _("The '--remote-region' and '--remote-image' options are required when using '--method=web-download'")
                raise exceptions.CommandError(msg)
        else:
            if parsed_args.remote_region:
                msg = _("The '--remote-region' option is only supported when using '--method=glance-download'")
                raise exceptions.CommandError(msg)
            if parsed_args.remote_image:
                msg = _("The '--remote-image' option is only supported when using '--method=glance-download'")
                raise exceptions.CommandError(msg)
            if parsed_args.remote_service_interface:
                msg = _("The '--remote-service-interface' option is only supported when using '--method=glance-download'")
                raise exceptions.CommandError(msg)
        if parsed_args.import_method == 'copy-image':
            if not (parsed_args.stores or parsed_args.all_stores):
                msg = _("The '--stores' or '--all-stores' options are required when using '--method=copy-image'")
                raise exceptions.CommandError(msg)
        image = image_client.find_image(parsed_args.image, ignore_missing=False)
        if not image.container_format and (not image.disk_format):
            msg = _("The 'container_format' and 'disk_format' properties must be set on an image before it can be imported")
            raise exceptions.CommandError(msg)
        if parsed_args.import_method == 'glance-direct':
            if image.status != 'uploading':
                msg = _("The 'glance-direct' import method can only be used with an image in status 'uploading'")
                raise exceptions.CommandError(msg)
        elif parsed_args.import_method == 'web-download':
            if image.status != 'queued':
                msg = _("The 'web-download' import method can only be used with an image in status 'queued'")
                raise exceptions.CommandError(msg)
        elif parsed_args.import_method == 'copy-image':
            if image.status != 'active':
                msg = _("The 'copy-image' import method can only be used with an image in status 'active'")
                raise exceptions.CommandError(msg)
        image_client.import_image(image, method=parsed_args.import_method, uri=parsed_args.uri, remote_region=parsed_args.remote_region, remote_image_id=parsed_args.remote_image, remote_service_interface=parsed_args.remote_service_interface, stores=parsed_args.stores, all_stores=parsed_args.all_stores, all_stores_must_succeed=not parsed_args.allow_failure)
        info = _format_image(image)
        return zip(*sorted(info.items()))