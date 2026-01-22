import os
import hashlib
from typing import Any, Dict, List, Optional
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
class K8SClient:
    """A Client class for K8S modules.

    This class has the primary purpose to proxy the kubernetes client and resource objects.
    If there is a need for other methods or attributes to be proxied, they can be added here.
    """
    K8S_SERVER_DRY_RUN = 'All'

    def __init__(self, configuration, client, dry_run: bool=False) -> None:
        self.configuration = configuration
        self.client = client
        self.dry_run = dry_run

    @property
    def resources(self) -> List[Any]:
        return self.client.resources

    def _find_resource_with_prefix(self, prefix: str, kind: str, api_version: str) -> Resource:
        for attribute in ['kind', 'name', 'singular_name']:
            try:
                return self.client.resources.get(**{'prefix': prefix, 'api_version': api_version, attribute: kind})
            except (ResourceNotFoundError, ResourceNotUniqueError):
                pass
        return self.client.resources.get(prefix=prefix, api_version=api_version, short_names=[kind])

    def resource(self, kind: str, api_version: str) -> Resource:
        """Fetch a kubernetes client resource.

        This will attempt to find a kubernetes resource trying, in order, kind,
        name, singular_name and short_names.
        """
        try:
            if api_version == 'v1':
                return self._find_resource_with_prefix('api', kind, api_version)
        except ResourceNotFoundError:
            pass
        return self._find_resource_with_prefix(None, kind, api_version)

    def _ensure_dry_run(self, params: Dict) -> Dict:
        if self.dry_run:
            params['dry_run'] = self.K8S_SERVER_DRY_RUN
        return params

    def validate(self, resource, version: Optional[str]=None, strict: Optional[bool]=False):
        return self.client.validate(resource, version, strict)

    def get(self, resource, **params):
        return resource.get(**params)

    def delete(self, resource, **params):
        return resource.delete(**self._ensure_dry_run(params))

    def apply(self, resource, definition, namespace, **params):
        return resource.apply(definition, namespace=namespace, **self._ensure_dry_run(params))

    def create(self, resource, definition, **params):
        return resource.create(definition, **self._ensure_dry_run(params))

    def replace(self, resource, definition, **params):
        return resource.replace(definition, **self._ensure_dry_run(params))

    def patch(self, resource, definition, **params):
        return resource.patch(definition, **self._ensure_dry_run(params))