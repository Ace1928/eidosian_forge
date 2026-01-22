from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
import six
def _SaveData(self):
    """Serializes data to the json file."""
    if not self._dirty:
        return
    files.WriteFileContents(self._last_update_check_file, json.dumps(self._data.ToDictionary()))
    self._dirty = False