from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
def _show_image(image):
    utils.print_dict(image._info)