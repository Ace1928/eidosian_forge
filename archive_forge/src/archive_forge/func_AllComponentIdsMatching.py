from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
def AllComponentIdsMatching(self, platform_filter):
    """Gets all components in the snapshot that match the given platform.

    Args:
      platform_filter: platforms.Platform, A platform the components must match.

    Returns:
      set(str), The matching component ids.
    """
    return set((c_id for c_id, component in six.iteritems(self.components) if component.platform.Matches(platform_filter)))