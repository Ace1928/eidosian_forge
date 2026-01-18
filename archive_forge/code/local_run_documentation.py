from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai.custom_jobs import flags
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.custom_jobs import validation
from googlecloudsdk.command_lib.ai.docker import build as docker_builder
from googlecloudsdk.command_lib.ai.docker import run as docker_runner
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
Run a custom training locally.

  Packages your training code into a Docker image and executes it locally.
  