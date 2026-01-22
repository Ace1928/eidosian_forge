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
def ConsumerClosureForComponents(self, component_ids, platform_filter=None):
    """Gets all the components that depend on any of the given ids.

    Args:
      component_ids: list of str, The ids of the components to get the consumers
        of.
      platform_filter: platforms.Platform, A platform that components must
        match to be pulled into the consumer closure.

    Returns:
      set of str, All component ids that are in the consumer closure, including
      the given components.
    """
    component_filter = lambda c: c.platform.Matches(platform_filter)
    return self._ClosureFor(component_ids, self.__consumers, component_filter)