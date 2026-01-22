from the command line arguments and returns a list of URLs to be given to the
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import io
import os
import sys
import textwrap
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.ml_engine import uploads
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
from six.moves import map
from setuptools import setup, find_packages
def GetStagingLocation(job_id=None, staging_bucket=None, job_dir=None):
    """Get the appropriate staging location for the job given the arguments."""
    staging_location = None
    if staging_bucket:
        staging_location = storage_util.ObjectReference.FromBucketRef(staging_bucket, job_id)
    elif job_dir:
        staging_location = storage_util.ObjectReference.FromName(job_dir.bucket, '/'.join([f for f in [job_dir.name.rstrip('/'), 'packages'] if f]))
    return staging_location