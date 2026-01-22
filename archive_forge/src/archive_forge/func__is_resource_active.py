from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def _is_resource_active(resource, dead_states):
    if dead_states is None:
        dead_states = DEAD_STATES
    if 'lifecycle_state' not in resource.attribute_map:
        return True
    return resource.lifecycle_state not in dead_states