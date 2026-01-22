from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import shutil
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@staticmethod
def BuildAndStoreFlexTemplateImage(image_gcr_path, flex_template_base_image, jar_paths, py_paths, go_binary_path, env, sdk_language, gcs_log_dir):
    """Builds the flex template docker container image and stores it in GCR.

    Args:
      image_gcr_path: GCR location to store the flex template container image.
      flex_template_base_image: SDK version or base image to use.
      jar_paths: List of jar paths to pipelines and dependencies.
      py_paths: List of python paths to pipelines and dependencies.
      go_binary_path: Path to compiled Go pipeline binary.
      env: Dictionary of env variables to set in the container image.
      sdk_language: SDK language of the flex template.
      gcs_log_dir: Path to Google Cloud Storage directory to store build logs.

    Returns:
      True if container is built and store successfully.

    Raises:
      ValueError: If the parameters values are invalid.
    """
    Templates.__ValidateFlexTemplateEnv(env, sdk_language)
    with files.TemporaryDirectory() as temp_dir:
        log.status.Print('Copying files to a temp directory {}'.format(temp_dir))
        pipeline_files = []
        paths = jar_paths
        if py_paths:
            paths = py_paths
        elif go_binary_path:
            paths = [go_binary_path]
        for path in paths:
            absl_path = os.path.abspath(path)
            if os.path.isfile(absl_path):
                shutil.copy2(absl_path, temp_dir)
            else:
                shutil.copytree(absl_path, '{}/{}'.format(temp_dir, os.path.basename(absl_path)))
            pipeline_files.append(os.path.split(absl_path)[1])
        log.status.Print('Generating dockerfile to build the flex template container image...')
        dockerfile_contents = Templates.BuildDockerfile(flex_template_base_image, pipeline_files, env, sdk_language)
        dockerfile_path = os.path.join(temp_dir, 'Dockerfile')
        files.WriteFileContents(dockerfile_path, dockerfile_contents)
        log.status.Print('Generated Dockerfile. Contents: {}'.format(dockerfile_contents))
        messages = cloudbuild_util.GetMessagesModule()
        build_config = submit_util.CreateBuildConfig(image_gcr_path, no_cache=False, messages=messages, substitutions=None, arg_config='cloudbuild.yaml', is_specified_source=True, no_source=False, source=temp_dir, gcs_source_staging_dir=None, ignore_file=None, arg_gcs_log_dir=gcs_log_dir, arg_machine_type=None, arg_disk_size=None, arg_worker_pool=None, arg_dir=None, arg_revision=None, arg_git_source_dir=None, arg_git_source_revision=None, arg_service_account=None, buildpack=None)
        log.status.Print('Pushing flex template container image to GCR...')
        submit_util.Build(messages, False, build_config)
        return True