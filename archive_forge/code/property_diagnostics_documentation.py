from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.diagnostics import check_base
from googlecloudsdk.core.diagnostics import diagnostic_base
import six
Run hidden property check.

    Args:
      first_run: bool, True if first time this has been run this invocation.

    Returns:
      A tuple of (check_base.Result, fixer) where fixer is a function that can
        be used to fix a failed check, or None if the check passed or failed
        with no applicable fix.
    