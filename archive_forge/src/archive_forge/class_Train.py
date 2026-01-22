from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Train(base.Command):
    """Submit an AI Platform training job."""
    _SUPPORT_TPU_TF_VERSION = False

    @classmethod
    def Args(cls, parser):
        _AddSubmitTrainingArgs(parser)
        flags.AddCustomContainerFlags(parser, support_tpu_tf_version=cls._SUPPORT_TPU_TF_VERSION)
        flags.AddKmsKeyFlag(parser, 'job')
        parser.display_info.AddFormat(jobs_util.JOB_FORMAT)

    def Run(self, args):
        stream_logs = jobs_util.GetStreamLogs(args.async_, args.stream_logs)
        scale_tier = jobs_util.ScaleTierFlagMap().GetEnumForChoice(args.scale_tier)
        scale_tier_name = scale_tier.name if scale_tier else None
        jobs_client = jobs.JobsClient()
        labels = jobs_util.ParseCreateLabels(jobs_client, args)
        custom_container_config = jobs_util.TrainingCustomInputServerConfig.FromArgs(args, self._SUPPORT_TPU_TF_VERSION)
        custom_container_config.ValidateConfig()
        job = jobs_util.SubmitTraining(jobs_client, args.job, job_dir=args.job_dir, staging_bucket=args.staging_bucket, packages=args.packages, package_path=args.package_path, scale_tier=scale_tier_name, config=args.config, module_name=args.module_name, runtime_version=args.runtime_version, python_version=args.python_version, network=args.network if hasattr(args, 'network') else None, service_account=args.service_account, labels=labels, stream_logs=stream_logs, user_args=args.user_args, kms_key=_GetAndValidateKmsKey(args), custom_train_server_config=custom_container_config, enable_web_access=args.enable_web_access)
        if stream_logs and job.state is not job.StateValueValuesEnum.SUCCEEDED:
            self.exit_code = 1
        return job