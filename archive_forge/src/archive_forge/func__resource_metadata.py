import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
from oslo_serialization import jsonutils
from urllib import request
from heatclient.common import format_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
def _resource_metadata(heat_client, args):
    fields = {'stack_id': args.stack, 'resource_name': args.resource}
    try:
        metadata = heat_client.resources.metadata(**fields)
    except heat_exc.HTTPNotFound:
        raise exc.CommandError(_('Stack %(stack)s or resource %(resource)s not found.') % {'stack': args.stack, 'resource': args.resource})
    data = list(metadata.values())
    columns = list(metadata.keys())
    return (columns, data)