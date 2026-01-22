from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import OrderedDict
import json
import re
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core.util import files
import six
def _PopSegments(parts, is_params):
    if parts:
        while parts[-1].startswith('{') == is_params and parts[-1].endswith('}') == is_params:
            parts.pop()
            if not parts:
                break