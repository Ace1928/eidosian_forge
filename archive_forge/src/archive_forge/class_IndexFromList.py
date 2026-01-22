from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_image_list
class IndexFromList(docker_image_list.Delegate):
    """This compatibility interface serves an Image Index from a Manifest List."""

    def __init__(self, image, recursive=True):
        """Constructor.

    Args:
      image: a DockerImageList on which __enter__ has already been called.
      recursive: whether to recursively convert child manifests to OCI types.
    """
        super(IndexFromList, self).__init__(image)
        self._recursive = recursive

    def manifest(self):
        """Override."""
        manifest = json.loads(self._image.manifest())
        manifest['mediaType'] = docker_http.OCI_IMAGE_INDEX_MIME
        return json.dumps(manifest, sort_keys=True)

    def media_type(self):
        """Override."""
        return docker_http.OCI_IMAGE_INDEX_MIME

    def __enter__(self):
        if not self._recursive:
            return self
        converted = []
        for platform, child in self._image:
            if isinstance(child, docker_image_list.DockerImageList):
                with IndexFromList(child) as index:
                    converted.append((platform, index))
            else:
                assert isinstance(child, docker_image.DockerImage)
                with OCIFromV22(child) as oci:
                    converted.append((platform, oci))
        with docker_image_list.FromList(converted) as index:
            self._image = index
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Override."""
        pass