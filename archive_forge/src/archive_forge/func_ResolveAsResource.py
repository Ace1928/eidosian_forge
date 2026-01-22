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
def ResolveAsResource(self, args, api_resource_registry, default_scope=None, scope_lister=None, with_project=True, source_project=None):
    """Resolve this resource against the arguments.

    Args:
      args: Namespace, argparse.Namespace.
      api_resource_registry: instance of core.resources.Registry.
      default_scope: ScopeEnum, ZONE, REGION, GLOBAL, or None when resolving
          name and scope was not specified use this as default. If there is
          exactly one possible scope it will be used, there is no need to
          specify default_scope.
      scope_lister: func(scope, underspecified_names), a callback which returns
        list of items (with 'name' attribute) for given scope.
      with_project: indicates whether or not project is associated. It should be
        False for flexible resource APIs.
      source_project: indicates whether or not a project is specified. It could
        be other projects. If it is None, then it will use the current project
        if with_project is true
    Returns:
      Resource reference or list of references if plural.
    """
    names = self._GetResourceNames(args)
    resource_scope, scope_value = self.scopes.SpecifiedByArgs(args)
    if resource_scope is None and self.scope_flags_usage == ScopeFlagsUsage.DONT_USE_SCOPE_FLAGS:
        resource_scope, scope_value = self.scopes.SpecifiedByValue(names[0])
    if resource_scope is not None:
        resource_scope = resource_scope.scope_enum
        if not self.required and (not names):
            if self.scopes.flag_prefix:
                flag = '--{0}-{1}'.format(self.scopes.flag_prefix, resource_scope.flag_name)
            else:
                flag = '--' + resource_scope
            raise exceptions.Error("Can't specify {0} without specifying resource via {1}".format(flag, self.name))
    refs = self._resource_resolver.ResolveResources(names, resource_scope, scope_value, api_resource_registry, default_scope, scope_lister, with_project=with_project, source_project=source_project)
    if self.plural:
        return refs
    if refs:
        return refs[0]
    return None