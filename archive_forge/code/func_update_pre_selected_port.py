import logging
import os
from typing import Dict, List, Optional
import ray._private.ray_constants as ray_constants
from ray._private.utils import (
def update_pre_selected_port(self):
    """Update the pre-selected port information

        Returns:
            The dictionary mapping of component -> ports.
        """

    def wrap_port(port):
        if port is None or port == 0:
            return []
        else:
            return [port]
    pre_selected_ports = {'gcs': wrap_port(self.redis_port), 'object_manager': wrap_port(self.object_manager_port), 'node_manager': wrap_port(self.node_manager_port), 'gcs_server': wrap_port(self.gcs_server_port), 'client_server': wrap_port(self.ray_client_server_port), 'dashboard': wrap_port(self.dashboard_port), 'dashboard_agent_grpc': wrap_port(self.metrics_agent_port), 'dashboard_agent_http': wrap_port(self.dashboard_agent_listen_port), 'dashboard_grpc': wrap_port(self.dashboard_grpc_port), 'runtime_env_agent': wrap_port(self.runtime_env_agent_port), 'metrics_export': wrap_port(self.metrics_export_port)}
    redis_shard_ports = self.redis_shard_ports
    if redis_shard_ports is None:
        redis_shard_ports = []
    pre_selected_ports['redis_shards'] = redis_shard_ports
    if self.worker_port_list is None:
        if self.min_worker_port is not None and self.max_worker_port is not None:
            pre_selected_ports['worker_ports'] = list(range(self.min_worker_port, self.max_worker_port + 1))
        else:
            pre_selected_ports['worker_ports'] = []
    else:
        pre_selected_ports['worker_ports'] = [int(port) for port in self.worker_port_list.split(',')]
    self.reserved_ports = set()
    for comp, port_list in pre_selected_ports.items():
        for port in port_list:
            if port in self.reserved_ports:
                raise ValueError(f'Ray component {comp} is trying to use a port number {port} that is used by other components.\nPort information: {self._format_ports(pre_selected_ports)}\nIf you allocate ports, please make sure the same port is not used by multiple components.')
            self.reserved_ports.add(port)