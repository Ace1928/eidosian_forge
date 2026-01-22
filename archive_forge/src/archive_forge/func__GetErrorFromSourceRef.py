from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetErrorFromSourceRef(obj, error_source_refs):
    """Helper function to get the actual error from the errorSourceRefs field.

  Args:
    obj: The RepoSync|RootSync object.
    error_source_refs: The errorSourceRefs value

  Returns:
    A list containing error values from the errorSourceRefs
  """
    errs = []
    for ref in error_source_refs:
        path = ref.split('.')
        err = _GetPathValue(obj, path)
        if err:
            errs.extend(err)
    return errs