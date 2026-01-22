import logging
import os
from oslo_middleware.healthcheck import opts
from oslo_middleware.healthcheck import pluginbase
@staticmethod
def _iter_paths_ports(paths):
    for port_path in paths:
        port_path = port_path.strip()
        if port_path:
            port, path = port_path.split(':', 1)
            port = int(port)
            yield (port, path)