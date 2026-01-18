from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.meta import generate_config_command
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
Generate declarative config commands with surface specs and tests.