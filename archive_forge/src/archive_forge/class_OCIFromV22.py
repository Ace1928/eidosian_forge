from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_image_list
class OCIFromV22(docker_image.Delegate):
    """This compatibility interface serves an OCI image from a v2_2 image."""

    def manifest(self):
        """Override."""
        manifest = json.loads(self._image.manifest())
        manifest['mediaType'] = docker_http.OCI_MANIFEST_MIME
        manifest['config']['mediaType'] = docker_http.OCI_CONFIG_JSON_MIME
        for layer in manifest['layers']:
            layer['mediaType'] = docker_http.OCI_LAYER_MIME
        return json.dumps(manifest, sort_keys=True)

    def media_type(self):
        """Override."""
        return docker_http.OCI_MANIFEST_MIME

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Override."""
        pass