from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.plugin_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException
from ansible_collections.community.library_inventory_filtering_v1.plugins.plugin_utils.inventory_filter import parse_filters, filter_host
def _slugify(self, value):
    return 'docker_%s' % re.sub('[^\\w-]', '_', value).lower().lstrip('_')