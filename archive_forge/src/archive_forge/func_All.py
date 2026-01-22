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
@staticmethod
def All():
    """Gets all the different states.

    Returns:
      list(ComponentStateTuple), All the states.
    """
    return [ComponentState.UPDATE_AVAILABLE, ComponentState.REMOVED, ComponentState.NEW, ComponentState.UP_TO_DATE]