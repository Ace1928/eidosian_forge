from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import enum
import functools
import re
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.api_lib.compute.regions import service as regions_service
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import scope_prompter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.util import text
import six
class ResourceArgScope(object):
    """Facilitates mapping of scope, flag and collection."""

    def __init__(self, scope, flag_prefix, collection):
        self.scope_enum = scope
        if flag_prefix:
            flag_prefix = flag_prefix.replace('-', '_')
            if scope is compute_scope.ScopeEnum.GLOBAL:
                self.flag_name = scope.flag_name + '_' + flag_prefix
            else:
                self.flag_name = flag_prefix + '_' + scope.flag_name
        else:
            self.flag_name = scope.flag_name
        self.flag = '--' + self.flag_name.replace('_', '-')
        self.collection = collection