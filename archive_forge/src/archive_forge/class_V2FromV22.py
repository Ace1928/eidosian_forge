from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util as v2_util
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
class V2FromV22(v2_image.DockerImage):
    """This compatibility interface serves the v2 interface from a v2_2 image."""

    def __init__(self, v2_2_img):
        """Constructor.

    Args:
      v2_2_img: a v2_2 DockerImage on which __enter__ has already been called.

    Raises:
      ValueError: an incorrectly typed argument was supplied.
    """
        self._v2_2_image = v2_2_img
        self._ProcessImage()

    def _ProcessImage(self):
        """Constructs schema 1 manifest from schema 2 manifest and config file."""
        manifest_schema2 = json.loads(self._v2_2_image.manifest())
        raw_config = self._v2_2_image.config_file()
        config = json.loads(raw_config)
        parent = ''
        histories = config.get('history', {})
        layer_count = len(histories)
        v2_layer_index = 0
        layers = manifest_schema2.get('layers', {})
        fs_layers = []
        v1_histories = []
        for v1_layer_index, history in enumerate(histories):
            digest, media_type, v2_layer_index = self._GetSchema1LayerDigest(history, layers, v1_layer_index, v2_layer_index)
            if v1_layer_index != layer_count - 1:
                layer_id = self._GenerateV1LayerId(digest, parent)
                v1_compatibility = self._BuildV1Compatibility(layer_id, parent, history)
            else:
                layer_id = self._GenerateV1LayerId(digest, parent, raw_config)
                v1_compatibility = self._BuildV1CompatibilityForTopLayer(layer_id, parent, history, config)
            parent = layer_id
            fs_layers = [{'blobSum': digest, 'mediaType': media_type}] + fs_layers
            v1_histories = [{'v1Compatibility': v1_compatibility}] + v1_histories
        manifest_schema1 = {'schemaVersion': 1, 'name': 'unused', 'tag': 'unused', 'fsLayers': fs_layers, 'history': v1_histories}
        if 'architecture' in config:
            manifest_schema1['architecture'] = config['architecture']
        self._manifest = v2_util.Sign(json.dumps(manifest_schema1, sort_keys=True))

    def _GenerateV1LayerId(self, digest, parent, raw_config=None):
        parts = digest.rsplit(':', 1)
        if len(parts) != 2:
            raise BadDigestException('Invalid Digest: %s, must be in algorithm : blobSumHex format.' % digest)
        data = parts[1] + ' ' + parent
        if raw_config:
            data += ' ' + raw_config
        return docker_digest.SHA256(data.encode('utf8'), '')

    def _BuildV1Compatibility(self, layer_id, parent, history):
        v1_compatibility = {'id': layer_id}
        if parent:
            v1_compatibility['parent'] = parent
        if 'empty_layer' in history:
            v1_compatibility['throwaway'] = True
        if 'created_by' in history:
            v1_compatibility['container_config'] = {'Cmd': [history['created_by']]}
        for key in ['created', 'comment', 'author']:
            if key in history:
                v1_compatibility[key] = history[key]
        return json.dumps(v1_compatibility, sort_keys=True)

    def _BuildV1CompatibilityForTopLayer(self, layer_id, parent, history, config):
        v1_compatibility = {'id': layer_id}
        if parent:
            v1_compatibility['parent'] = parent
        if 'empty_layer' in history:
            v1_compatibility['throwaway'] = True
        for key in ['architecture', 'container', 'docker_version', 'os', 'config', 'container_config', 'created']:
            if key in config:
                v1_compatibility[key] = config[key]
        return json.dumps(v1_compatibility, sort_keys=True)

    def _GetSchema1LayerDigest(self, history, layers, v1_layer_index, v2_layer_index):
        if 'empty_layer' in history:
            return (EMPTY_TAR_DIGEST, docker_http.LAYER_MIME, v2_layer_index)
        else:
            return (layers[v2_layer_index]['digest'], layers[v2_layer_index]['mediaType'], v2_layer_index + 1)

    def manifest(self):
        """Override."""
        return self._manifest

    def uncompressed_blob(self, digest):
        """Override."""
        if digest == EMPTY_TAR_DIGEST:
            return super(V2FromV22, self).uncompressed_blob(EMPTY_TAR_DIGEST)
        return self._v2_2_image.uncompressed_blob(digest)

    def diff_id(self, digest):
        """Gets v22 diff_id."""
        return self._v2_2_image.digest_to_diff_id(digest)

    def blob(self, digest):
        """Override."""
        if digest == EMPTY_TAR_DIGEST:
            return EMPTY_TAR_BYTES
        return self._v2_2_image.blob(digest)

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass