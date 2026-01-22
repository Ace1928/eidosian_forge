from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags
from googlecloudsdk.command_lib.builds import staging_bucket_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def SetSource(build_config, messages, is_specified_source, no_source, source, gcs_source_staging_dir, arg_dir, arg_revision, arg_git_source_dir, arg_git_source_revision, ignore_file, hide_logs=False, build_region=cloudbuild_util.DEFAULT_REGION, arg_bucket_behavior=None):
    """Set the source for the build config."""
    default_gcs_source = False
    default_bucket_name = None
    default_bucket_location = cloudbuild_util.DEFAULT_REGION
    if gcs_source_staging_dir is None:
        default_gcs_source = True
        if build_region != cloudbuild_util.DEFAULT_REGION and arg_bucket_behavior is not None and (flags.GetDefaultBuckestBehavior(arg_bucket_behavior) == messages.BuildOptions.DefaultLogsBucketBehaviorValueValuesEnum.REGIONAL_USER_OWNED_BUCKET):
            default_bucket_location = build_region
            default_bucket_name = staging_bucket_util.GetDefaultRegionalStagingBucket(build_region)
        else:
            default_bucket_name = staging_bucket_util.GetDefaultStagingBucket()
        gcs_source_staging_dir = 'gs://{}/source'.format(default_bucket_name)
    gcs_client = storage_api.StorageClient()
    if not is_specified_source and no_source:
        source = None
    if source:
        if any((source.startswith(x) for x in ['http://', 'https://'])):
            build_config.source = messages.Source(gitSource=messages.GitSource(url=source, dir=arg_git_source_dir, revision=arg_git_source_revision))
            return build_config
        if re.match('projects/.*/locations/.*/connections/.*/repositories/.*', source):
            build_config.source = messages.Source(connectedRepository=messages.ConnectedRepository(repository=source, dir=arg_dir, revision=arg_revision))
            return build_config
        suffix = '.tgz'
        if source.startswith('gs://') or os.path.isfile(source):
            _, suffix = os.path.splitext(source)
        staged_object = '{stamp}-{uuid}{suffix}'.format(stamp=times.GetTimeStampFromDateTime(times.Now()), uuid=uuid.uuid4().hex, suffix=suffix)
        gcs_source_staging_dir = resources.REGISTRY.Parse(gcs_source_staging_dir, collection='storage.objects')
        try:
            if default_bucket_location == cloudbuild_util.DEFAULT_REGION:
                gcs_client.CreateBucketIfNotExists(gcs_source_staging_dir.bucket, check_ownership=default_gcs_source)
            else:
                gcs_client.CreateBucketIfNotExists(gcs_source_staging_dir.bucket, location=default_bucket_location, check_ownership=default_gcs_source)
        except api_exceptions.HttpForbiddenError:
            raise BucketForbiddenError('The user is forbidden from accessing the bucket [{}]. Please check your organization\'s policy or if the user has the "serviceusage.services.use" permission. Giving the user Owner, Editor, or Viewer roles may also fix this issue. Alternatively, use the --no-source option and access your source code via a different method.'.format(gcs_source_staging_dir.bucket))
        except storage_api.BucketInWrongProjectError:
            raise c_exceptions.RequiredArgumentException('gcs-source-staging-dir', 'A bucket with name {} already exists and is owned by another project. Specify a bucket using --gcs-source-staging-dir.'.format(default_bucket_name))
        if gcs_source_staging_dir.object:
            staged_object = gcs_source_staging_dir.object + '/' + staged_object
        gcs_source_staging = resources.REGISTRY.Create(collection='storage.objects', bucket=gcs_source_staging_dir.bucket, object=staged_object)
        staged_source_obj = staging_bucket_util.Upload(source, gcs_source_staging, gcs_client, ignore_file=ignore_file, hide_logs=hide_logs)
        build_config.source = messages.Source(storageSource=messages.StorageSource(bucket=staged_source_obj.bucket, object=staged_source_obj.name, generation=staged_source_obj.generation))
    elif not no_source:
        raise c_exceptions.InvalidArgumentException('--no-source', 'To omit source, use the --no-source flag.')
    return build_config