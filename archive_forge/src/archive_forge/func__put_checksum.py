from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_creds as v1_creds
from containerregistry.client.v1 import docker_http
from containerregistry.client.v1 import docker_image
import httplib2
import six.moves.http_client
def _put_checksum(self, image, layer_id):
    """Upload the checksum for a single layer."""
    pass