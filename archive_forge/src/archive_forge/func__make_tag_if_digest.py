import argparse
import logging
import sys
import tarfile
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list as image_list
from containerregistry.client.v2_2 import save
from containerregistry.client.v2_2 import v2_compat
from containerregistry.tools import logging_setup
from containerregistry.tools import patched
from containerregistry.tools import platform_args
from containerregistry.transport import retry
from containerregistry.transport import transport_pool
import httplib2
def _make_tag_if_digest(name):
    if isinstance(name, docker_name.Tag):
        return name
    return docker_name.Tag('{repo}:{tag}'.format(repo=str(name.as_repository()), tag=_DEFAULT_TAG))