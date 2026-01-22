from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
def __AddFlag(self, flag, name):
    choices = 'bool'
    if flag.choices:
        choices = sorted(flag.choices)
        if choices == ['false', 'true']:
            choices = 'bool'
    elif flag.nargs != 0:
        choices = 'dynamic' if getattr(flag, 'completer', None) else 'value'
    self.flags[name] = choices