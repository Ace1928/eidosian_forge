from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.command_lib.logs import stream
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_prep
from googlecloudsdk.command_lib.ml_engine import log_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def SubmitTraining(jobs_client, job, job_dir=None, staging_bucket=None, packages=None, package_path=None, scale_tier=None, config=None, module_name=None, runtime_version=None, network=None, service_account=None, python_version=None, stream_logs=None, user_args=None, labels=None, kms_key=None, custom_train_server_config=None, enable_web_access=None):
    """Submit a training job."""
    region = properties.VALUES.compute.region.Get(required=True)
    staging_location = jobs_prep.GetStagingLocation(staging_bucket=staging_bucket, job_id=job, job_dir=job_dir)
    try:
        uris = jobs_prep.UploadPythonPackages(packages=packages, package_path=package_path, staging_location=staging_location)
    except jobs_prep.NoStagingLocationError:
        raise flags.ArgumentError('If local packages are provided, the `--staging-bucket` or `--job-dir` flag must be given.')
    log.debug('Using {0} as trainer uris'.format(uris))
    scale_tier_enum = jobs_client.training_input_class.ScaleTierValueValuesEnum
    scale_tier = scale_tier_enum(scale_tier) if scale_tier else None
    try:
        job = jobs_client.BuildTrainingJob(path=config, module_name=module_name, job_name=job, trainer_uri=uris, region=region, job_dir=job_dir.ToUrl() if job_dir else None, scale_tier=scale_tier, user_args=user_args, runtime_version=runtime_version, network=network, service_account=service_account, python_version=python_version, labels=labels, kms_key=kms_key, custom_train_server_config=custom_train_server_config, enable_web_access=enable_web_access)
    except jobs_prep.NoStagingLocationError:
        raise flags.ArgumentError('If `--package-path` is not specified, at least one Python package must be specified via `--packages`.')
    project_ref = resources.REGISTRY.Parse(properties.VALUES.core.project.Get(required=True), collection='ml.projects')
    job = jobs_client.Create(project_ref, job)
    if not stream_logs:
        PrintSubmitFollowUp(job.jobId, print_follow_up_message=True)
        return job
    else:
        PrintSubmitFollowUp(job.jobId, print_follow_up_message=False)
    log_fetcher = stream.LogFetcher(filters=log_utils.LogFilters(job.jobId), polling_interval=properties.VALUES.ml_engine.polling_interval.GetInt(), continue_interval=_CONTINUE_INTERVAL, continue_func=log_utils.MakeContinueFunction(job.jobId))
    printer = resource_printer.Printer(log_utils.LOG_FORMAT, out=log.err)
    with execution_utils.RaisesKeyboardInterrupt():
        try:
            printer.Print(log_utils.SplitMultiline(log_fetcher.YieldLogs()))
        except KeyboardInterrupt:
            log.status.Print('Received keyboard interrupt.\n')
            log.status.Print(_FOLLOW_UP_MESSAGE.format(job_id=job.jobId, project=project_ref.Name()))
        except exceptions.HttpError as err:
            log.status.Print('Polling logs failed:\n{}\n'.format(six.text_type(err)))
            log.info('Failure details:', exc_info=True)
            log.status.Print(_FOLLOW_UP_MESSAGE.format(job_id=job.jobId, project=project_ref.Name()))
    job_ref = resources.REGISTRY.Parse(job.jobId, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection='ml.projects.jobs')
    job = jobs_client.Get(job_ref)
    return job