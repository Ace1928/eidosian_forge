from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag

        Lookup and inspect each image name found in the names parameter.

        :returns array of image dictionaries
        