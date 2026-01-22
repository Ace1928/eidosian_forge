from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import operator
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import text
def _PromptSuggestedScopeChoice(resource_name, underspecified_names, scope_enum, suggested_resource):
    if scope_enum == compute_scope.ScopeEnum.GLOBAL:
        log.status.Print('No scope specified. Using [{0}] for {1}: [{2}].'.format(scope_enum.flag_name, text.Pluralize(len(underspecified_names), resource_name), ', '.join(underspecified_names)))
    else:
        log.status.Print('No {0} specified. Using {0} [{1}] for {2}: [{3}].'.format(scope_enum.flag_name, suggested_resource, text.Pluralize(len(underspecified_names), resource_name), ', '.join(underspecified_names)))