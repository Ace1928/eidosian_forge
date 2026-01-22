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