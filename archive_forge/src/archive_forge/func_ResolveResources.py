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
def ResolveResources(self, names, resource_scope, scope_value, api_resource_registry, default_scope=None, scope_lister=None, with_project=True, source_project=None):
    """Resolve this resource against the arguments.

    Args:
      names: list of str, list of resource names
      resource_scope: ScopeEnum, kind of scope of resources; if this is not None
                   scope_value should be name of scope of type specified by this
                   argument. If this is None scope_value should be None, in that
                   case if prompting is possible user will be prompted to
                   select scope (if prompting is forbidden it will raise an
                   exception).
      scope_value: ScopeEnum, scope of resources; if this is not None
                   resource_scope should be type of scope specified by this
                   argument. If this is None resource_scope should be None, in
                   that case if prompting is possible user will be prompted to
                   select scope (if prompting is forbidden it will raise an
                   exception).
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
    Raises:
      BadArgumentException: when names is not a list or default_scope is not one
          of the configured scopes.
      UnderSpecifiedResourceError: if it was not possible to resolve given names
          as resources references.
    """
    self._ValidateNames(names)
    self._ValidateDefaultScope(default_scope)
    if resource_scope is not None:
        resource_scope = self.scopes[resource_scope]
    if default_scope is not None:
        default_scope = self.scopes[default_scope]
    if source_project is not None:
        source_project_ref = api_resource_registry.Parse(source_project, collection='compute.projects')
        source_project = source_project_ref.Name()
    project = source_project or properties.VALUES.core.project.GetOrFail()
    if with_project:
        params = {'project': project}
    else:
        params = {}
    if scope_value is None:
        resource_scope = self.scopes.GetImplicitScope(default_scope)
    resource_scope_param = self._GetResourceScopeParam(resource_scope, scope_value, project, api_resource_registry, with_project=with_project)
    if resource_scope_param is not None:
        params[resource_scope.scope_enum.param_name] = resource_scope_param
    collection = resource_scope and resource_scope.collection
    refs, underspecified_names = self._GetRefsAndUnderspecifiedNames(names, params, collection, scope_value is not None, api_resource_registry)
    if underspecified_names and len(self.scopes) > 1:
        self._ResolveMultiScope(with_project, project, underspecified_names, api_resource_registry, refs)
    self._ResolveUnderspecifiedNames(underspecified_names, default_scope, scope_lister, project, api_resource_registry, with_project=with_project)
    refs = [ref[0] for ref in refs]
    expected_collections = [scope.collection for scope in self.scopes]
    for ref in refs:
        if ref.Collection() not in expected_collections:
            raise resources.WrongResourceCollectionException(expected=','.join(expected_collections), got=ref.Collection(), path=ref.SelfLink())
    return refs