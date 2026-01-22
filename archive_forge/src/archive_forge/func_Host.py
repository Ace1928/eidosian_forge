import os
import json
from shutil import copyfile, rmtree
from docker.tls import TLSConfig
from docker.errors import ContextException
from docker.context.config import get_meta_dir
from docker.context.config import get_meta_file
from docker.context.config import get_tls_dir
from docker.context.config import get_context_host
@property
def Host(self):
    if not self.orchestrator or self.orchestrator == 'swarm':
        endpoint = self.endpoints.get('docker', None)
        if endpoint:
            return endpoint.get('Host', None)
        return None
    return self.endpoints[self.orchestrator].get('Host', None)