from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import posixpath
import re
import textwrap
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils
from googlecloudsdk.core import log
from six.moves import shlex_quote
def _IsVertexTrainingPrebuiltImage(image_name):
    """Checks whether the image is pre-built by Vertex AI training."""
    prebuilt_image_name_regex = '^(us|europe|asia)-docker.pkg.dev/vertex-ai/training/(tf|scikit-learn|pytorch|xgboost)-.+$'
    return re.fullmatch(prebuilt_image_name_regex, image_name) is not None