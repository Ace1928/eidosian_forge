from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _get_attrs_compute(client_manager, parsed_args):
    attrs = {}
    if parsed_args.name is not None:
        attrs['name'] = parsed_args.name
    if parsed_args.share:
        attrs['share_subnet'] = True
    if parsed_args.no_share:
        attrs['share_subnet'] = False
    if parsed_args.subnet is not None:
        attrs['subnet'] = parsed_args.subnet
    return attrs