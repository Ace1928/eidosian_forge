from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.config.virtualenv import util
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
class VirtualEnvInfo(object):

    def __init__(self, python_version, modules, enabled):
        self.python_version = python_version
        self.modules = modules
        self.enabled = enabled