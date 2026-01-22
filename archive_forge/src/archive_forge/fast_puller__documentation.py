import argparse
import logging
import sys
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
This package pulls images from a Docker Registry.

Unlike docker_puller the format this uses is proprietary.
