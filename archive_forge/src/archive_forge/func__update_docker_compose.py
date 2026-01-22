import argparse
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import yaml
def _update_docker_compose(docker_compose_path: str, project_name: str, status: Optional[Dict[str, Any]]) -> bool:
    docker_compose_config = _read_yaml(docker_compose_path)
    if not docker_compose_config:
        print('Docker compose currently empty')
        return False
    cmd = ['up', '-d']
    if status and len(status) > 0:
        cmd += ['--no-recreate']
    shutdown = False
    if not docker_compose_config['services']:
        print('Shutting down nodes')
        cmd = ['down']
        shutdown = True
    try:
        for node_id, node_conf in docker_compose_config['services'].items():
            for volume_mount in node_conf['volumes']:
                host_dir, container_dir = volume_mount.split(':', maxsplit=1)
                if container_dir == '/cluster/node' and (not os.path.exists(host_dir)):
                    os.makedirs(host_dir, 493, exist_ok=True)
        subprocess.check_output(['docker', 'compose', '-f', docker_compose_path, '-p', project_name] + cmd + ['--remove-orphans'])
    except Exception as e:
        print(f'Ran into error when updating docker compose: {e}')
    return shutdown