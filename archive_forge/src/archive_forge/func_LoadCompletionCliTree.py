from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
def LoadCompletionCliTree():
    """Loads and returns the static completion CLI tree."""
    try:
        sys_path = sys.path[:]
        sys.path.append(_GetCompletionCliTreeDir())
        import gcloud_completions
        tree = gcloud_completions.STATIC_COMPLETION_CLI_TREE
    except ImportError:
        raise CannotHandleCompletionError('Cannot find static completion CLI tree module.')
    finally:
        sys.path = sys_path
    return tree