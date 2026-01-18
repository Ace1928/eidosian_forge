from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import shutil
import tempfile
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.app.runtimes import fingerprinter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import deployables
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.command_lib.app import output_helpers
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.api import appinfo
from ruamel import yaml
import six
Generate missing configuration files for a source directory.