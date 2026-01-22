from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _LabelMatched(obj, selector_map):
    """Checked if the given object matched with the label selectors."""
    if not obj:
        return False
    if not selector_map:
        return True
    labels = _GetPathValue(obj, ['metadata', 'labels'])
    if not labels:
        return False
    for key in selector_map:
        value = selector_map[key]
        if key not in labels or labels[key] != value:
            return False
    return True