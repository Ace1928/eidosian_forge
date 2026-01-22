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
def BuildAndPushDockerImage(project, service, upload_dir, source_files, version_id, code_bucket_ref, gcr_domain, runtime_builder_strategy=runtime_builders.RuntimeBuilderStrategy.NEVER, parallel_build=False, use_flex_with_buildpacks=False):
    """Builds and pushes a set of docker images.

  Args:
    project: str, The project being deployed to.
    service: ServiceYamlInfo, The parsed service config.
    upload_dir: str, path to the service's upload directory
    source_files: [str], relative paths to upload.
    version_id: The version id to deploy these services under.
    code_bucket_ref: The reference to the GCS bucket where the source will be
      uploaded.
    gcr_domain: str, Cloud Registry domain, determines the physical location
      of the image. E.g. `us.gcr.io`.
    runtime_builder_strategy: runtime_builders.RuntimeBuilderStrategy, whether
      to use the new CloudBuild-based runtime builders (alternative is old
      externalized runtimes).
    parallel_build: bool, if True, enable parallel build and deploy.
    use_flex_with_buildpacks: bool, if true, use the build-image and
      run-image built through buildpacks.

  Returns:
    BuildArtifact, Representing the pushed container image or in-progress build.

  Raises:
    DockerfileError: if a Dockerfile is present, but the runtime is not
      "custom".
    NoDockerfileError: Raised if a user didn't supply a Dockerfile and chose a
      custom runtime.
    UnsatisfiedRequirementsError: Raised if the code in the directory doesn't
      satisfy the requirements of the specified runtime type.
    ValueError: if an unrecognized runtime_builder_strategy is given
  """
    needs_dockerfile = _NeedsDockerfile(service, upload_dir)
    use_runtime_builders = ShouldUseRuntimeBuilders(service, runtime_builder_strategy, needs_dockerfile)
    if not service.RequiresImage():
        return None
    log.status.Print('Building and pushing image for service [{service}]'.format(service=service.module))
    gen_files = dict(_GetSourceContextsForUpload(upload_dir))
    if needs_dockerfile and (not use_runtime_builders):
        gen_files.update(_GetDockerfiles(service, upload_dir))
    image = docker_image.Image(dockerfile_dir=upload_dir, repo=_GetImageName(project, service.module, version_id, gcr_domain), nocache=False, tag=config.DOCKER_IMAGE_TAG)
    metrics.CustomTimedEvent(metric_names.CLOUDBUILD_UPLOAD_START)
    object_ref = storage_util.ObjectReference.FromBucketRef(code_bucket_ref, image.tagged_repo)
    relative_yaml_path = _GetYamlPath(upload_dir, service.file, service.parsed.skip_files, gen_files)
    try:
        cloud_build.UploadSource(upload_dir, source_files, object_ref, gen_files=gen_files)
    except (OSError, IOError) as err:
        if platforms.OperatingSystem.IsWindows():
            if err.filename and len(err.filename) > _WINDOWS_MAX_PATH:
                raise WindowMaxPathError(err.filename)
        raise
    metrics.CustomTimedEvent(metric_names.CLOUDBUILD_UPLOAD)
    if use_runtime_builders:
        builder_reference = runtime_builders.FromServiceInfo(service, upload_dir, use_flex_with_buildpacks)
        log.info('Using runtime builder [%s]', builder_reference.build_file_uri)
        builder_reference.WarnIfDeprecated()
        yaml_path = util.ConvertToPosixPath(relative_yaml_path)
        substitute = {'_OUTPUT_IMAGE': image.tagged_repo, '_GAE_APPLICATION_YAML_PATH': yaml_path}
        if use_flex_with_buildpacks:
            python_version = yaml_parsing.GetRuntimeConfigAttr(service.parsed, 'python_version')
            if yaml_parsing.GetRuntimeConfigAttr(service.parsed, 'python_version'):
                substitute['_GOOGLE_RUNTIME_VERSION'] = python_version
        build = builder_reference.LoadCloudBuild(substitute)
    else:
        build = cloud_build.GetDefaultBuild(image.tagged_repo)
    build = cloud_build.FixUpBuild(build, object_ref)
    return _SubmitBuild(build, image, project, parallel_build)