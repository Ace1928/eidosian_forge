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
def _diff_id_to_digest(self, diff_id):
    for this_digest, this_diff_id in six.moves.zip(self.fs_layers(), self.diff_ids()):
        if this_diff_id == diff_id:
            return this_digest
    raise ValueError('Unmatched "diff_id": "%s"' % diff_id)