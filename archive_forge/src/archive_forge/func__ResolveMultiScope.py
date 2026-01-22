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