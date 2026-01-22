from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import matrix_creator_common
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six
def _BuildAppReference(self, filename):
    """Builds either a FileReference or an AppBundle message for a file."""
    if filename.endswith('.aab'):
        return (None, self._messages.AppBundle(bundleLocation=self._BuildFileReference(os.path.basename(filename))))
    else:
        return (self._BuildFileReference(os.path.basename(filename)), None)