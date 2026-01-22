import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from heatclient.common import format_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
def _list_resourcetypes(heat_client, parsed_args):
    resource_types = heat_client.resource_types.list(filters=heat_utils.format_parameters(parsed_args.filter), with_description=parsed_args.long)
    if parsed_args.long:
        columns = ['Resource Type', 'Description']
        rows = sorted(([r.resource_type, r.description] for r in resource_types))
    else:
        columns = ['Resource Type']
        rows = sorted(([r.resource_type] for r in resource_types))
    return (columns, rows)