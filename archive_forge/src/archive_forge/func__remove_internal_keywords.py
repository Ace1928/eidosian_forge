from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _remove_internal_keywords(self, resource):
    items = ['kind', 'generation', 'selfLink', 'poolReference', 'offset', 'datagroupReference']
    for item in items:
        try:
            del resource[item]
        except KeyError:
            pass