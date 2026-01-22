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
class ResourceArgument(object):
    """Encapsulates concept of compute resource as command line argument.

  Basic Usage:
    class MyCommand(base.Command):
      _BACKEND_SERVICE_ARG = flags.ResourceArgument(
          resource_name='backend service',
          completer=compute_completers.BackendServiceCompleter,
          regional_collection='compute.regionBackendServices',
          global_collection='compute.backendServices')
      _INSTANCE_GROUP_ARG = flags.ResourceArgument(
          resource_name='instance group',
          completer=compute_completers.InstanceGroupsCompleter,
          zonal_collection='compute.instanceGroups',)

      @staticmethod
      def Args(parser):
        MyCommand._BACKEND_SERVICE_ARG.AddArgument(parser)
        MyCommand._INSTANCE_GROUP_ARG.AddArgument(parser)

      def Run(args):
        api_resource_registry = resources.REGISTRY.CloneAndSwitch(
            api_tools_client)
        backend_service_ref = _BACKEND_SERVICE_ARG.ResolveAsResource(
            args, api_resource_registry, default_scope=flags.ScopeEnum.GLOBAL)
        instance_group_ref = _INSTANCE_GROUP_ARG.ResolveAsResource(
            args, api_resource_registry, default_scope=flags.ScopeEnum.ZONE)
        ...

    In the above example the following five arguments/flags will be defined:
      NAME - positional for backend service
      --region REGION to qualify backend service
      --global  to qualify backend service
      --instance-group INSTANCE_GROUP name for the instance group
      --instance-group-zone INSTANCE_GROUP_ZONE further qualifies instance group

    More generally this construct can simultaneously support global, regional
    and zonal qualifiers (or any combination of) for each resource.
  """

    def __init__(self, name=None, resource_name=None, completer=None, plural=False, required=True, zonal_collection=None, regional_collection=None, global_collection=None, global_help_text=None, region_explanation=None, region_help_text=None, region_hidden=False, zone_explanation=None, zone_help_text=None, zone_hidden=False, short_help=None, detailed_help=None, custom_plural=None, scope_flags_usage=ScopeFlagsUsage.GENERATE_DEDICATED_SCOPE_FLAGS):
        """Constructor.

    Args:
      name: str, argument name.
      resource_name: str, human readable name for resources eg "instance group".
      completer: completion_cache.Completer, The completer class type.
      plural: bool, whether to accept multiple values.
      required: bool, whether this argument is required.
      zonal_collection: str, include zone flag and use this collection
                             to resolve it.
      regional_collection: str, include region flag and use this collection
                                to resolve it.
      global_collection: str, if also zonal and/or regional adds global flag
                              and uses this collection to resolve as
                              global resource.
      global_help_text: str, if provided, global flag help text will be
                             overridden with this value.
      region_explanation: str, long help that will be given for region flag,
                               empty by default.
      region_help_text: str, if provided, region flag help text will be
                             overridden with this value.
      region_hidden: bool, Hide region in help if True.
      zone_explanation: str, long help that will be given for zone flag, empty
                             by default.
      zone_help_text: str, if provided, zone flag help text will be overridden
                           with this value.
      zone_hidden: bool, Hide zone in help if True.
      short_help: str, help for the flag being added, if not provided help text
                       will be 'The name[s] of the ${resource_name}[s].'.
      detailed_help: str, detailed help for the flag being added, if not
                          provided there will be no detailed help for the flag.
      custom_plural: str, If plural is True then this string will be used as
                          plural resource name.
      scope_flags_usage: ScopeFlagsUsage, when set to
                                  USE_EXISTING_SCOPE_FLAGS, already existing
                                  zone and/or region flags will be used for
                                  this argument,
                                  GENERATE_DEDICATED_SCOPE_FLAGS, new scope
                                  flags will be created,
                                  DONT_USE_SCOPE_FLAGS to not generate
                                  additional flags and use single argument for
                                  all scopes.

    Raises:
      exceptions.Error: if there some inconsistency in arguments.
    """
        self.name_arg = name or 'name'
        self._short_help = short_help
        self._detailed_help = detailed_help
        self.scope_flags_usage = scope_flags_usage
        if self.name_arg.startswith('--'):
            self.is_flag = True
            self.name = self.name_arg[2:].replace('-', '_')
            flag_prefix = None if self.scope_flags_usage == ScopeFlagsUsage.USE_EXISTING_SCOPE_FLAGS else self.name_arg[2:]
            self.scopes = ResourceArgScopes(flag_prefix=flag_prefix)
        else:
            self.scopes = ResourceArgScopes(flag_prefix=None)
            self.name = self.name_arg
        self.resource_name = resource_name
        self.completer = completer
        self.plural = plural
        self.custom_plural = custom_plural
        self.required = required
        if not (zonal_collection or regional_collection or global_collection):
            raise exceptions.Error('Must specify at least one resource type zonal, regional or global')
        if zonal_collection:
            self.scopes.AddScope(compute_scope.ScopeEnum.ZONE, collection=zonal_collection)
        if regional_collection:
            self.scopes.AddScope(compute_scope.ScopeEnum.REGION, collection=regional_collection)
        if global_collection:
            self.scopes.AddScope(compute_scope.ScopeEnum.GLOBAL, collection=global_collection)
        self._global_help_text = global_help_text
        self._region_explanation = region_explanation or ''
        self._region_help_text = region_help_text
        self._region_hidden = region_hidden
        self._zone_explanation = zone_explanation or ''
        self._zone_help_text = zone_help_text
        self._zone_hidden = zone_hidden
        self._resource_resolver = ResourceResolver(self.scopes, resource_name)

    def AddArgument(self, parser, mutex_group=None, operation_type='operate on', cust_metavar=None, category=None, scope_required=False):
        """Add this set of arguments to argparse parser."""
        params = dict(metavar=cust_metavar if cust_metavar else self.name.upper(), completer=self.completer)
        if self._detailed_help:
            params['help'] = self._detailed_help
        elif self._short_help:
            params['help'] = self._short_help
        else:
            params['help'] = 'Name{} of the {} to {}.'.format('s' if self.plural else '', text.Pluralize(int(self.plural) + 1, self.resource_name or '', self.custom_plural), operation_type)
            if self.name.startswith('instance'):
                params['help'] += " For details on valid instance names, refer to the criteria documented under the field 'name' at: https://cloud.google.com/compute/docs/reference/rest/v1/instances"
            if self.name == 'DISK_NAME' and operation_type == 'create':
                params['help'] += ' For details on the naming convention for this resource, refer to: https://cloud.google.com/compute/docs/naming-resources'
        if self.name_arg.startswith('--'):
            params['required'] = self.required
            if not self.required:
                params['category'] = category
            if self.plural:
                params['type'] = arg_parsers.ArgList(min_length=1)
        elif self.required:
            if self.plural:
                params['nargs'] = '+'
        else:
            params['nargs'] = '*' if self.plural else '?'
        (mutex_group or parser).add_argument(self.name_arg, **params)
        if self.scope_flags_usage != ScopeFlagsUsage.GENERATE_DEDICATED_SCOPE_FLAGS:
            return
        if len(self.scopes) > 1:
            scope = parser.add_group(mutex=True, category=category, required=scope_required)
        else:
            scope = parser
        if compute_scope.ScopeEnum.ZONE in self.scopes:
            AddZoneFlag(scope, flag_prefix=self.scopes.flag_prefix, resource_type=self.resource_name, operation_type=operation_type, explanation=self._zone_explanation, help_text=self._zone_help_text, hidden=self._zone_hidden, plural=self.plural, custom_plural=self.custom_plural)
        if compute_scope.ScopeEnum.REGION in self.scopes:
            AddRegionFlag(scope, flag_prefix=self.scopes.flag_prefix, resource_type=self.resource_name, operation_type=operation_type, explanation=self._region_explanation, help_text=self._region_help_text, hidden=self._region_hidden, plural=self.plural, custom_plural=self.custom_plural)
        if compute_scope.ScopeEnum.GLOBAL in self.scopes and len(self.scopes) > 1:
            if not self.plural:
                resource_mention = '{} is'.format(self.resource_name)
            elif self.plural and (not self.custom_plural):
                resource_mention = '{}s are'.format(self.resource_name)
            else:
                resource_mention = '{} are'.format(self.custom_plural)
            scope.add_argument(self.scopes[compute_scope.ScopeEnum.GLOBAL].flag, action='store_true', default=None, help=self._global_help_text or 'If set, the {0} global.'.format(resource_mention))

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

    def _GetResourceNames(self, args):
        """Return list of resource names specified by args."""
        if self.plural:
            return getattr(args, self.name)
        name_value = getattr(args, self.name)
        if name_value is not None:
            return [name_value]
        return []