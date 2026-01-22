from oslo_log import log as logging
from saharaclient.osc.v1 import data_sources as ds_v1
class ListDataSources(ds_v1.ListDataSources):
    """Lists data sources"""
    log = logging.getLogger(__name__ + '.ListDataSources')