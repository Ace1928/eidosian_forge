from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import openssl_encryption_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as core_encoding
from googlecloudsdk.core.util import files
Returns the decrypted password from the data in the serial port.