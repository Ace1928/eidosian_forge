import json
from troveclient import base
from troveclient import common
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
class MgmtDatastoreVersions(base.ManagerWithFind):
    """Manage :class:`DatastoreVersion` resources."""
    resource_class = datastores.DatastoreVersion

    def list(self, limit=None, marker=None):
        """List all datastore versions."""
        return self._paginated('/mgmt/datastore-versions', 'versions', limit, marker)

    def get(self, datastore_version_id):
        """Get details of a datastore version."""
        return self._get('/mgmt/datastore-versions/%s' % datastore_version_id, 'version')

    def create(self, name, datastore_name, datastore_manager, image, packages=None, active='true', default='false', image_tags=[], version=None):
        """Create a new datastore version."""
        packages = packages or []
        body = {'version': {'name': name, 'datastore_name': datastore_name, 'datastore_manager': datastore_manager, 'image_tags': image_tags, 'packages': packages, 'active': json.loads(active), 'default': json.loads(default)}}
        if image:
            body['version']['image'] = image
        if version:
            body['version']['version'] = version
        return self._create('/mgmt/datastore-versions', body, None, True)

    def edit(self, datastore_version_id, datastore_manager=None, image=None, packages=None, active=None, default=None, image_tags=None, name=None):
        """Update a datastore-version."""
        packages = packages or []
        body = {}
        if datastore_manager is not None:
            body['datastore_manager'] = datastore_manager
        if image is not None:
            body['image'] = image
        if packages:
            body['packages'] = packages
        if active is not None:
            body['active'] = json.loads(active)
        if default is not None:
            body['default'] = json.loads(default)
        if image_tags is not None:
            body['image_tags'] = image_tags
        if name:
            body['name'] = name
        url = '/mgmt/datastore-versions/%s' % datastore_version_id
        resp, body = self.api.client.patch(url, body=body)
        common.check_for_exceptions(resp, body, url)

    def delete(self, datastore_version_id):
        """Delete a datastore version."""
        url = '/mgmt/datastore-versions/%s' % datastore_version_id
        resp, body = self.api.client.delete(url)
        common.check_for_exceptions(resp, body, url)