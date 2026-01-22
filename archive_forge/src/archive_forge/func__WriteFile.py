from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _WriteFile(file_path, file_contents, enable_overwrites):
    if not os.path.exists(file_path) or enable_overwrites:
        with files.FileWriter(file_path, create_path=True) as f:
            f.write(file_contents)