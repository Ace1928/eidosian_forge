import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from heatclient.common import format_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
def _show_resourcetype(heat_client, parsed_args):
    try:
        if parsed_args.template_type:
            template_type = parsed_args.template_type.lower()
            if template_type not in ('hot', 'cfn'):
                raise exc.CommandError(_('Template type invalid: %s') % parsed_args.template_type)
            fields = {'resource_type': parsed_args.resource_type, 'template_type': template_type}
            data = heat_client.resource_types.generate_template(**fields)
        else:
            data = heat_client.resource_types.get(parsed_args.resource_type, parsed_args.long)
    except heat_exc.HTTPNotFound:
        raise exc.CommandError(_('Resource type not found: %s') % parsed_args.resource_type)
    rows = list(data.values())
    columns = list(data.keys())
    return (columns, rows)