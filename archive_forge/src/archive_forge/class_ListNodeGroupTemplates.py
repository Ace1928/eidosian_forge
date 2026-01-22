import sys
from osc_lib import utils as osc_utils
from saharaclient.osc import utils
from saharaclient.osc.v1 import node_group_templates as ngt_v1
class ListNodeGroupTemplates(ngt_v1.ListNodeGroupTemplates, utils.NodeGroupTemplatesUtils):
    """Lists node group templates"""

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        return self._list_take_action(client, self.app, parsed_args)