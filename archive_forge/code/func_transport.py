import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
@property
def transport(self):
    """The transport way to be sent data to server
        (default is sync).
        """
    return self._transport