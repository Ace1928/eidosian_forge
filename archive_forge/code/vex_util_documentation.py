from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import re
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.core import log
from  googlecloudsdk.core.util.files import FileReader
Parse GCR URL.

  Args:
    url: gcr url for version, tag or whole image

  Returns:
    strings of project, image url and version url

  Raises:
    ar_exceptions.InvalidInputValueError: If user input is invalid.
  