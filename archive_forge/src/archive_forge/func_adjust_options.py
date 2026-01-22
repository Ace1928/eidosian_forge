from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def adjust_options(opts, states):
    adjust_data_volumes(opts, states)
    adjust_nics(opts, states)