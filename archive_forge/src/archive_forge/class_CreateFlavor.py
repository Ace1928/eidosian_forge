import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class CreateFlavor(cli.CreateFlavor):
    """Create a pool flavor"""

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)' % parsed_args)
        client = self.app.client_manager.messaging
        kwargs = {}
        if parsed_args.capabilities != {}:
            raise AttributeError('<--capabilities> option is only             available in client api version < 2')
        pool_list = None
        if parsed_args.pool_list:
            pool_list = parsed_args.pool_list.split(',')
        data = client.flavor(parsed_args.flavor_name, pool_list=pool_list, **kwargs)
        columns = ('Name', 'Pool list', 'Capabilities')
        return (columns, utils.get_item_properties(data, columns))