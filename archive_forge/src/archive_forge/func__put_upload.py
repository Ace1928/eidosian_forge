from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import concurrent.futures
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_image_list as image_list
import httplib2
import six.moves.http_client
import six.moves.urllib.parse
def _put_upload(self, image, digest):
    mounted, location = self._start_upload(digest, self._mount)
    if mounted:
        logging.info('Layer %s mounted.', digest)
        return
    location = self._add_digest(location, digest)
    self._transport.Request(location, method='PUT', body=self._get_blob(image, digest), accepted_codes=[six.moves.http_client.CREATED])