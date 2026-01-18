import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
@property
def view_name_to_data_map(self):
    """Map with all view data objects
        that will be sent to Prometheus
        """
    return self._view_name_to_data_map