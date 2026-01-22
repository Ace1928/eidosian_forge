from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util as v2_util
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
def _GetSchema1LayerDigest(self, history, layers, v1_layer_index, v2_layer_index):
    if 'empty_layer' in history:
        return (EMPTY_TAR_DIGEST, docker_http.LAYER_MIME, v2_layer_index)
    else:
        return (layers[v2_layer_index]['digest'], layers[v2_layer_index]['mediaType'], v2_layer_index + 1)