from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import retry
import six
def WaitForResourceDeletion(request_method, resource_ref, message, timeout_s=60, poll_period_s=5):
    """Poll Dataproc resource until it no longer exists."""
    with progress_tracker.ProgressTracker(message, autotick=True):
        start_time = time.time()
        while timeout_s > time.time() - start_time:
            try:
                request_method(resource_ref)
            except apitools_exceptions.HttpNotFoundError:
                return
            except apitools_exceptions.HttpError as error:
                log.debug('Get request for [{0}] failed:\n{1}', resource_ref, error)
                if IsClientHttpException(error):
                    raise
            time.sleep(poll_period_s)
    raise exceptions.OperationTimeoutError('Deleting resource [{0}] timed out.'.format(resource_ref))