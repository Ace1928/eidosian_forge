from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.diagnose import external_helper
from googlecloudsdk.command_lib.compute.diagnose import internal_helpers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
Returns a list of instances according to the flags.