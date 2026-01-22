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
def _ComputeState(self):
    """Returns the component state."""
    if self.__current is None:
        return ComponentState.NEW
    elif self.__latest is None:
        return ComponentState.REMOVED
    elif self.__latest.version.build_number > self.__current.version.build_number:
        return ComponentState.UPDATE_AVAILABLE
    elif self.__latest.version.build_number < self.__current.version.build_number:
        if self.__latest.data is None and self.__current.data is None:
            return ComponentState.UP_TO_DATE
        elif bool(self.__latest.data) ^ bool(self.__current.data):
            return ComponentState.UPDATE_AVAILABLE
        elif self.__latest.data.contents_checksum != self.__current.data.contents_checksum:
            return ComponentState.UPDATE_AVAILABLE
    return ComponentState.UP_TO_DATE