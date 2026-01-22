from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import constants
from googlecloudsdk.core.resource import resource_expr_rewrite
import six
def _RewriteKey(self, key):
    for regex, replacement in six.iteritems(self._KEY_MAPPING):
        if re.match(regex, key):
            return re.sub(regex, replacement, key)
    return key