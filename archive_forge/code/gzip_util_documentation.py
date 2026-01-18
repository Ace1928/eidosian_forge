from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gzip
import os
import shutil
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
Determines if file qualifies for in-flight gzip encoding.