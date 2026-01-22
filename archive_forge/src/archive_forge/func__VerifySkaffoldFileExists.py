from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import datetime
import io
import os.path
import shutil
import tarfile
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import snapshot
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.code.cloud import cloudrun
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import staging_bucket_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _VerifySkaffoldFileExists(source, skaffold_file):
    """Checks that the specified source contains a skaffold configuration file."""
    if not skaffold_file:
        skaffold_file = 'skaffold.yaml'
    if source.startswith('gs://'):
        log.status.Print('Skipping skaffold file check. Reason: source is not a local archive or directory')
    elif not os.path.exists(source):
        raise c_exceptions.BadFileException('could not find source [{src}]'.format(src=source))
    elif os.path.isfile(source):
        _VerifySkaffoldIsInArchive(source, skaffold_file)
    else:
        _VerifySkaffoldIsInFolder(source, skaffold_file)