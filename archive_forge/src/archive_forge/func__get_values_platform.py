from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils.module_container.base import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _get_values_platform(module, container, api_version, options, image, host_info):
    if image and (image.get('Os') or image.get('Architecture') or image.get('Variant')):
        return {'platform': compose_platform_string(os=image.get('Os'), arch=image.get('Architecture'), variant=image.get('Variant'), daemon_os=host_info.get('OSType') if host_info else None, daemon_arch=host_info.get('Architecture') if host_info else None)}
    return {'platform': container.get('Platform')}