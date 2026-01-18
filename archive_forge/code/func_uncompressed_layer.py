from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import tarfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
import httplib2
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import six.moves.http_client
def uncompressed_layer(self, diff_id):
    if diff_id in self._uncompressed_layer_to_filename:
        with io.open(self._uncompressed_layer_to_filename[diff_id], 'rb') as reader:
            return reader.read()
    if self._legacy_base and diff_id in self._legacy_base.diff_ids():
        return self._legacy_base.uncompressed_layer(diff_id)
    return super(FromDisk, self).uncompressed_layer(diff_id)