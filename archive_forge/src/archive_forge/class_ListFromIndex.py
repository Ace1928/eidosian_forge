from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_image_list
class ListFromIndex(docker_image_list.Delegate):
    """This compatibility interface serves a Manifest List from an Image Index."""

    def __init__(self, image, recursive=True):
        """Constructor.

    Args:
      image: a DockerImageList on which __enter__ has already been called.
      recursive: whether to recursively convert child manifests to Docker types.
    """
        super(ListFromIndex, self).__init__(image)
        self._recursive = recursive

    def manifest(self):
        """Override."""
        manifest = json.loads(self._image.manifest())
        manifest['mediaType'] = docker_http.MANIFEST_LIST_MIME
        return json.dumps(manifest, sort_keys=True)

    def media_type(self):
        """Override."""
        return docker_http.MANIFEST_LIST_MIME

    def __enter__(self):
        if not self._recursive:
            return self
        converted = []
        for platform, child in self._image:
            if isinstance(child, docker_image_list.DockerImageList):
                with ListFromIndex(child) as image_list:
                    converted.append((platform, image_list))
            else:
                assert isinstance(child, docker_image.DockerImage)
                with V22FromOCI(child) as v22:
                    converted.append((platform, v22))
        with docker_image_list.FromList(converted) as image_list:
            self._image = image_list
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Override."""
        pass