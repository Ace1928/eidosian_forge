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
def _delete_config(heat_client, args):
    failure_count = 0
    for config_id in args.config:
        try:
            heat_client.software_configs.delete(config_id=config_id)
        except Exception as e:
            if isinstance(e, heat_exc.HTTPNotFound):
                print(_('Software config with ID %s not found') % config_id)
            failure_count += 1
            continue
    if failure_count:
        raise exc.CommandError(_('Unable to delete %(count)s of the %(total)s software configs.') % {'count': failure_count, 'total': len(args.config)})