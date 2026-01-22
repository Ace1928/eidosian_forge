from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import build as app_build
from googlecloudsdk.api_lib.app import cloud_build
from googlecloudsdk.api_lib.app import docker_image
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.api_lib.app.runtimes import fingerprinter
from googlecloudsdk.api_lib.cloudbuild import build as cloudbuild_build
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as s_exceptions
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.tools import context_util
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
def _GetSourceContextsForUpload(source_dir):
    """Gets source context file information.

  Args:
    source_dir: str, path to the service's source directory
  Returns:
    A dict of filename to (str) source context file contents.
  """
    source_contexts = {}
    m = 'Could not generate [{name}]: {error}\nStackdriver Debugger may not be configured or enabled on this application. See https://cloud.google.com/debugger/ for more information.'
    try:
        contexts = context_util.CalculateExtendedSourceContexts(source_dir)
    except context_util.GenerateSourceContextError as e:
        log.info(m.format(name=context_util.CONTEXT_FILENAME, error=e))
        return source_contexts
    try:
        context = context_util.BestSourceContext(contexts)
        source_contexts[context_util.CONTEXT_FILENAME] = six.text_type(json.dumps(context))
    except KeyError as e:
        log.info(m.format(name=context_util.CONTEXT_FILENAME, error=e))
    return source_contexts