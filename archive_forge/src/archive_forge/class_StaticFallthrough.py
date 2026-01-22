from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
class StaticFallthrough(Fallthrough):
    """Falls through to a hardcoded value."""

    def __init__(self, argument, value):
        super(StaticFallthrough, self).__init__('leave the argument unspecified for it to be chosen automatically')
        self._handled_fields = [argument]
        self.value = value

    def _Call(self, parsed_args):
        return self.value