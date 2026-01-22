from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
def _RunCheck(self, check, first_run=True):
    with progress_tracker.ProgressTracker('{0} {1}'.format('Checking' if first_run else 'Rechecking', check.issue)):
        result, fixer = check.Check(first_run=first_run)
    self._PrintResult(result)
    return (result, fixer)