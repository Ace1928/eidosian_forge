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
class MalformedSnapshotError(Error):
    """Error with the contents of the snapshot."""

    def __init__(self):
        super(MalformedSnapshotError, self).__init__('Failed to process component listing from server.')