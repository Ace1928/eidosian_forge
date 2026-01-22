from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
def RequestedScope(args):
    install = 'installation' if getattr(args, 'installation', False) else None
    scope_arg = getattr(args, 'scope', None)
    return properties.Scope.FromId(scope_arg or install)