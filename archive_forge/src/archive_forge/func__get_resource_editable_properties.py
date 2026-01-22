from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (Config, HwcClientException,
import re
def _get_resource_editable_properties(module):
    return {'display_name': module.params.get('display_name')}