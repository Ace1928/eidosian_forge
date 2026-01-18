from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import time
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
List all PyPI modules installed in an Airflow worker.

  ## EXAMPLES

    The following command:

    {command} myenv

    runs the "python -m pip list" command on a worker and returns the output.

    The following command:

    {command} myenv --tree

    runs the "python -m pipdeptree --warn" command on a worker and returns the
    output.
  