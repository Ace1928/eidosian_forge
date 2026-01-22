from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.run import run_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import sources
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _PrepareSubmitBuildRequest(tracker, build_image, build_source, build_pack, region, resource_ref, release_track, base_image):
    """Upload the provided build source and prepare submit build request."""
    messages = run_util.GetMessagesModule(release_track)
    tracker.StartStage(stages.UPLOAD_SOURCE)
    tracker.UpdateHeaderMessage('Uploading sources.')
    source = sources.Upload(build_source, region, resource_ref)
    tracker.CompleteStage(stages.UPLOAD_SOURCE)
    parent = 'projects/{project}/locations/{region}'.format(project=properties.VALUES.core.project.Get(required=True), region=region)
    storage_source = messages.GoogleCloudRunV2StorageSource(bucket=source.bucket, object=source.name, generation=source.generation)
    buildpack_build = None
    docker_build = None
    if build_pack:
        function_target = None
        envs = build_pack[0].get('envs', [])
        function_target_env = [x for x in envs if x.startswith('GOOGLE_FUNCTION_TARGET')]
        if function_target_env:
            function_target = function_target_env[0].split('=')[1]
        buildpack_build = messages.GoogleCloudRunV2BuildpacksBuild(baseImage=base_image, functionTarget=function_target)
    else:
        docker_build = messages.GoogleCloudRunV2DockerBuild()
    submit_build_request = messages.RunProjectsLocationsBuildsSubmitRequest(parent=parent, googleCloudRunV2SubmitBuildRequest=messages.GoogleCloudRunV2SubmitBuildRequest(storageSource=storage_source, imageUri=build_image, buildpackBuild=buildpack_build, dockerBuild=docker_build))
    return submit_build_request