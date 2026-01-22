from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
class ShowImage(command.ShowOne):
    """Describe a specific image"""
    log = logging.getLogger(__name__ + '.ShowImage')

    def get_parser(self, prog_name):
        parser = super(ShowImage, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of image to describe')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['id'] = parsed_args.uuid
        image = client.images.get(**opts)
        columns = _image_columns(image)
        return (columns, utils.get_item_properties(image, columns))