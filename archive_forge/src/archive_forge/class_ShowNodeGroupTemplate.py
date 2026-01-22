import sys
from osc_lib import utils as osc_utils
from saharaclient.osc import utils
from saharaclient.osc.v1 import node_group_templates as ngt_v1
class ShowNodeGroupTemplate(ngt_v1.ShowNodeGroupTemplate, utils.NodeGroupTemplatesUtils):
    """Display node group template details"""

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        data = utils.get_resource(client.node_group_templates, parsed_args.node_group_template).to_dict()
        _format_ngt_output(data)
        data = utils.prepare_data(data, NGT_FIELDS)
        return self.dict2columns(data)