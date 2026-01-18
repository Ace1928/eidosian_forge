from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import errno
import io
import json
import os
import tarfile
import concurrent.futures
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_image as v1_image
from containerregistry.client.v1 import save as v1_save
from containerregistry.client.v2 import v1_compat
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import v2_compat
import six
Creates a symbolic link dest pointing to source.

    Unlinks first to remove "old" layers if needed
    e.g., image A latest has layers 1, 2 and 3
    after a while it has layers 1, 2 and 3'.
    Since in both cases the layers are named 001, 002 and 003,
    unlinking promises the correct layers are linked in the image directory.

    Args:
      source: image directory source.
      dest: image directory destination.
    