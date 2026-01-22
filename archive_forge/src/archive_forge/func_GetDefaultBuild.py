from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gzip
import io
import operator
import os
import tarfile
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def GetDefaultBuild(output_image):
    """Get the default build for this runtime.

  This build just uses the latest docker builder image (location pulled from the
  app/container_builder_image property) to run a `docker build` with the given
  tag.

  Args:
    output_image: GCR location for the output docker image (e.g.
      `gcr.io/test-gae/hardcoded-output-tag`)

  Returns:
    Build, a CloudBuild Build message with the given steps (ready to be given to
      FixUpBuild).
  """
    messages = cloudbuild_util.GetMessagesModule()
    builder = properties.VALUES.app.container_builder_image.Get()
    log.debug('Using builder image: [{0}]'.format(builder))
    return messages.Build(steps=[messages.BuildStep(name=builder, args=['build', '-t', output_image, '.'])], images=[output_image])