from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import random
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list as v2_2_image_list
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import files
import requests
import six
from six.moves import urllib
def UploadSbomToGCS(source, artifact, sbom, gcs_path=None):
    """Upload an SBOM file onto the GCS bucket in the given project and location.

  Args:
    source: str, the SBOM file location.
    artifact: Artifact, the artifact metadata SBOM file generated from.
    sbom: SbomFile, metadata of the SBOM file.
    gcs_path: str, the GCS location for the SBOm file. If not provided, will use
      the default bucket path of the artifact.

  Returns:
    dest: str, the GCS storage path the file is copied to.
  """
    gcs_client = storage_api.StorageClient()
    if gcs_path:
        dest = _GetSbomGCSPath(gcs_path, artifact.resource_uri, sbom)
    else:
        project_num = project_util.GetProjectNumber(artifact.project)
        bucket_project = artifact.project
        bucket_location = artifact.location
        if bucket_location == 'europe':
            bucket_location = 'eu'
        default_bucket = _DefaultGCSBucketName(project_num, bucket_location)
        bucket_name = default_bucket
        use_backup_bucket = False
        try:
            gcs_client.CreateBucketIfNotExists(bucket=bucket_name, project=bucket_project, location=bucket_location, check_ownership=True)
        except storage_api.BucketInWrongProjectError:
            log.debug('The default bucket is in a wrong project.')
            use_backup_bucket = True
        except apitools_exceptions.HttpForbiddenError:
            log.debug('The default bucket cannot be accessed.')
            use_backup_bucket = True
        if use_backup_bucket:
            bucket_name = _FindAvailableGCSBucket(default_bucket, bucket_project, bucket_location)
        log.debug('Using bucket: {}'.format(bucket_name))
        dest = _GetSbomGCSPath(bucket_name, artifact.resource_uri, sbom)
    target_ref = storage_util.ObjectReference.FromUrl(dest)
    gcs_client.CopyFileToGCS(source, target_ref)
    return dest