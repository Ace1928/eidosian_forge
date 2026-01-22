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
def _GetSuggestedScopeValue(scope):
    if scope == compute_scope.ScopeEnum.ZONE:
        return _GetGCEZone()
    if scope == compute_scope.ScopeEnum.REGION:
        return _GetGCERegion()
    return True