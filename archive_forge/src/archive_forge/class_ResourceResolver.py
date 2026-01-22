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
class ResourceResolver(object):
    """Object responsible for resolving resources.

  There are two ways to build an instance of this object:
  1. Preferred when you don't have instance of ResourceArgScopes already built,
     using .FromMap static function. For example:

     resolver = ResourceResolver.FromMap(
         'instance',
         {compute_scope.ScopeEnum.ZONE: 'compute.instances'})

     where:
     - 'instance' is human readable name of the resource,
     - dictionary maps allowed scope (in this case only zone) to resource types
       in those scopes.
     - optional prefix of scope flags was skipped.

  2. Using constructor. Recommended only if you have instance of
     ResourceArgScopes available.

  Once you've built the resover you can use it to build resource references (and
  prompt for scope if it was not specified):

  resolver.ResolveResources(
        instance_name, compute_scope.ScopeEnum.ZONE,
        instance_zone, self.resources,
        scope_lister=flags.GetDefaultScopeLister(
            self.compute_client, self.project))

  will return a list of instances (of length 0 or 1 in this case, because we
  pass a name of single instance or None). It will prompt if and only if
  instance_name was not None but instance_zone was None.

  scope_lister is necessary for prompting.
  """

    def __init__(self, scopes, resource_name):
        """Initilize ResourceResolver instance.

    Prefer building with FromMap unless you have ResourceArgScopes object
    already built.

    Args:
      scopes: ResourceArgScopes, allowed scopes and resource types in those
              scopes.
      resource_name: str, human readable name for resources eg
                     "instance group".
    """
        self.scopes = scopes
        self.resource_name = resource_name

    @staticmethod
    def FromMap(resource_name, scopes_map, scope_flag_prefix=None):
        """Initilize ResourceResolver instance.

    Args:
      resource_name: str, human readable name for resources eg
                     "instance group".
      scopes_map: dict, with keys should be instances of ScopeEnum, values
              should be instances of ResourceArgScope.
      scope_flag_prefix: str, prefix of flags specyfying scope.
    Returns:
      New instance of ResourceResolver.
    """
        scopes = ResourceArgScopes(flag_prefix=scope_flag_prefix)
        for scope, resource in six.iteritems(scopes_map):
            scopes.AddScope(scope, resource)
        return ResourceResolver(scopes, resource_name)

    def _ValidateNames(self, names):
        if not isinstance(names, list):
            raise BadArgumentException("Expected names to be a list but it is '{0}'".format(names))

    def _ValidateDefaultScope(self, default_scope):
        if default_scope is not None and default_scope not in self.scopes:
            raise BadArgumentException('Unexpected value for default_scope {0}, expected None or {1}'.format(default_scope, ' or '.join([s.scope_enum.name for s in self.scopes])))

    def _GetResourceScopeParam(self, resource_scope, scope_value, project, api_resource_registry, with_project=True):
        """Gets the resource scope parameters."""
        if scope_value is not None:
            if resource_scope.scope_enum == compute_scope.ScopeEnum.GLOBAL:
                return None
            else:
                collection = compute_scope.ScopeEnum.CollectionForScope(resource_scope.scope_enum)
                if with_project:
                    return api_resource_registry.Parse(scope_value, params={'project': project}, collection=collection).Name()
                else:
                    return api_resource_registry.Parse(scope_value, params={}, collection=collection).Name()
        elif resource_scope and resource_scope.scope_enum != compute_scope.ScopeEnum.GLOBAL:
            return resource_scope.scope_enum.property_func

    def _GetRefsAndUnderspecifiedNames(self, names, params, collection, scope_defined, api_resource_registry):
        """Returns pair of lists: resolved references and unresolved names.

    Args:
      names: list of names to attempt resolving
      params: params given when attempting to resolve references
      collection: collection for the names
      scope_defined: bool, whether scope is known
      api_resource_registry: Registry object
    """
        refs = []
        underspecified_names = []
        for name in names:
            try:
                ref = [api_resource_registry.Parse(name, params=params, collection=collection, enforce_collection=False)]
            except (resources.UnknownCollectionException, resources.RequiredFieldOmittedException, properties.RequiredPropertyError):
                if scope_defined:
                    raise
                ref = [name]
                underspecified_names.append(ref)
            refs.append(ref)
        return (refs, underspecified_names)

    def _ResolveMultiScope(self, with_project, project, underspecified_names, api_resource_registry, refs):
        """Resolve argument against available scopes of the resource."""
        names = copy.deepcopy(underspecified_names)
        for scope in self.scopes:
            if with_project:
                params = {'project': project}
            else:
                params = {}
            params[scope.scope_enum.param_name] = scope.scope_enum.property_func
            for name in names:
                try:
                    ref = [api_resource_registry.Parse(name[0], params=params, collection=scope.collection, enforce_collection=False)]
                    refs.remove(name)
                    refs.append(ref)
                    underspecified_names.remove(name)
                except (resources.UnknownCollectionException, resources.RequiredFieldOmittedException, properties.RequiredPropertyError, ValueError):
                    continue

    def _ResolveUnderspecifiedNames(self, underspecified_names, default_scope, scope_lister, project, api_resource_registry, with_project=True):
        """Attempt to resolve scope for unresolved names.

    If unresolved_names was generated with _GetRefsAndUnderspecifiedNames
    changing them will change corresponding elements of refs list.

    Args:
      underspecified_names: list of one-items lists containing str
      default_scope: default scope for the resources
      scope_lister: callback used to list potential scopes for the resources
      project: str, id of the project
      api_resource_registry: resources Registry
      with_project: indicates whether or not project is associated. It should be
        False for flexible resource APIs

    Raises:
      UnderSpecifiedResourceError: when resource scope can't be resolved.
    """
        if not underspecified_names:
            return
        names = [n[0] for n in underspecified_names]
        if not console_io.CanPrompt():
            raise UnderSpecifiedResourceError(names, [s.flag for s in self.scopes])
        resource_scope_enum, scope_value = scope_prompter.PromptForScope(self.resource_name, names, [s.scope_enum for s in self.scopes], default_scope.scope_enum if default_scope is not None else None, scope_lister)
        if resource_scope_enum is None:
            raise UnderSpecifiedResourceError(names, [s.flag for s in self.scopes])
        resource_scope = self.scopes[resource_scope_enum]
        if with_project:
            params = {'project': project}
        else:
            params = {}
        if resource_scope.scope_enum != compute_scope.ScopeEnum.GLOBAL:
            params[resource_scope.scope_enum.param_name] = scope_value
        for name in underspecified_names:
            name[0] = api_resource_registry.Parse(name[0], params=params, collection=resource_scope.collection, enforce_collection=True)

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