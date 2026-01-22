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
class IncompatibleSchemaVersionError(Error):
    """Error for when we are unable to parse the new version of the snapshot."""

    def __init__(self, schema_version):
        super(IncompatibleSchemaVersionError, self).__init__('The latest version snapshot is incompatible with this installation.')
        self.schema_version = schema_version