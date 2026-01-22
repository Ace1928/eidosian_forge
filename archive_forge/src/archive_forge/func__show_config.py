import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient.common import template_format
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
def _show_config(heat_client, config_id, config_only):
    try:
        sc = heat_client.software_configs.get(config_id=config_id)
    except heat_exc.HTTPNotFound:
        raise exc.CommandError(_('Configuration not found: %s') % config_id)
    columns = None
    rows = None
    if config_only:
        print(sc.config)
    else:
        columns = ('id', 'name', 'group', 'config', 'inputs', 'outputs', 'options', 'creation_time')
        rows = utils.get_dict_properties(sc.to_dict(), columns)
    return (columns, rows)