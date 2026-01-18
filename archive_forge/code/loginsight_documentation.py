import json
import logging as log
from urllib import parse as urlparse
import netaddr
from oslo_concurrency.lockutils import synchronized
import requests
from osprofiler.drivers import base
from osprofiler import exc
Retrieves and parses trace data from Log Insight.

        :param base_id: Trace base ID
        