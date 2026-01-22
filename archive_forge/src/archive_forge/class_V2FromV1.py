from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v1 import docker_image as v1_image
from containerregistry.client.v2 import docker_digest
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util
from six.moves import zip  # pylint: disable=redefined-builtin
class V2FromV1(v2_image.DockerImage):
    """This compatibility interface serves the v2 interface from a v1 image."""

    def __init__(self, v1_img):
        """Constructor.

    Args:
      v1_img: a v1 DockerImage on which __enter__ has already been called.

    Raises:
      ValueError: an incorrectly typed argument was supplied.
    """
        self._v1_image = v1_img
        self._ProcessImage()

    def _ProcessImage(self):
        fs_layers = []
        self._layer_map = {}
        for layer_id in self._v1_image.ancestry(self._v1_image.top()):
            blob = self._v1_image.layer(layer_id)
            digest = docker_digest.SHA256(blob)
            fs_layers += [{'blobSum': digest}]
            self._layer_map[digest] = layer_id
        self._manifest = util.Sign(json.dumps({'schemaVersion': 1, 'name': 'unused', 'tag': 'unused', 'architecture': 'amd64', 'fsLayers': fs_layers, 'history': [{'v1Compatibility': self._v1_image.json(layer_id)} for layer_id in self._v1_image.ancestry(self._v1_image.top())]}, sort_keys=True))

    def manifest(self):
        """Override."""
        return self._manifest

    def uncompressed_blob(self, digest):
        """Override."""
        return self._v1_image.uncompressed_layer(self._layer_map[digest])

    def blob(self, digest):
        """Override."""
        return self._v1_image.layer(self._layer_map[digest])

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass