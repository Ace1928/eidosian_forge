import json
import logging
from http.client import RemoteDisconnected
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME
def _get_http_response(self, request):
    headers = {'Content-Type': 'application/json'}
    request_message = json.dumps(request).encode()
    http_coordinator_address = 'http://' + self.coordinator_address
    try:
        import requests
        from requests.exceptions import ConnectionError
        r = requests.get(http_coordinator_address, data=request_message, headers=headers, timeout=None)
    except (RemoteDisconnected, ConnectionError):
        logger.exception('Could not connect to: ' + http_coordinator_address + '. Did you run python coordinator_server.py' + ' --ips <list_of_node_ips> --port <PORT>?')
        raise
    except ImportError:
        logger.exception('Not all Ray Autoscaler dependencies were found. In Ray 1.4+, the Ray CLI, autoscaler, and dashboard will only be usable via `pip install "ray[default]"`. Please update your install command.')
        raise
    response = r.json()
    return response