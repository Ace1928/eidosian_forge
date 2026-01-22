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
def CheckMissingPlatformExecutable(self, component_ids, platform_filter=None):
    """Gets all the components that miss required platform-specific executables.

    Args:
      component_ids: list of str, The ids of the components to check for.
      platform_filter: platforms.Platform, A platform that components must
        match to be pulled into the dependency closure.

    Returns:
      set of str, All component ids that miss required platform-specific
        executables.
    """
    invalid_seeds = set()
    for c_id in component_ids:
        if c_id in self.components and (not self.components[c_id].platform.architectures) and (not self.components[c_id].platform.operating_systems) and self.components[c_id].platform_required:
            deps = self.DependencyClosureForComponents([c_id], platform_filter=platform_filter)
            qualified = [d for d in deps if str(d).startswith('{}-'.format(c_id))]
            if not qualified:
                invalid_seeds.add(c_id)
    return invalid_seeds