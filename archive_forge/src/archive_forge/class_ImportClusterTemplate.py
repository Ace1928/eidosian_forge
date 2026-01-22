from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import cluster_templates as ct_v1
class ImportClusterTemplate(ct_v1.ImportClusterTemplate):
    """Imports cluster template"""
    log = logging.getLogger(__name__ + '.ImportClusterTemplate')

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        data = self._take_action(client, parsed_args)
        _format_ct_output(self.app, data)
        data = utils.prepare_data(data, ct_v1.CT_FIELDS)
        return self.dict2columns(data)