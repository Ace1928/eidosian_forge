from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _environment_var_to_network_endpoints(endpoints):
    """Yields a dict with ip address and port."""
    for endpoint in endpoints.split(','):
        grpc_prefix = 'grpc://'
        if endpoint.startswith(grpc_prefix):
            endpoint = endpoint.split(grpc_prefix)[1]
        parts = endpoint.split(':')
        ip_address = parts[0]
        port = _DEFAULT_ENDPOINT_PORT
        if len(parts) > 1:
            port = parts[1]
        yield {'ipAddress': ip_address, 'port': port}