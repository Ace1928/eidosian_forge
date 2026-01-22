from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
class DeleteImage(command.Command):
    """Delete specified image from a host"""
    log = logging.getLogger(__name__ + '.DeleteImage')

    def get_parser(self, prog_name):
        parser = super(DeleteImage, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of image to describe')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['image_id'] = parsed_args.uuid
        try:
            client.images.delete(**opts)
            print(_('Request to delete image %s has been accepted.') % opts['image_id'])
        except Exception as e:
            print('Delete for image %(image)s failed: %(e)s' % {'image': opts['image_id'], 'e': e})