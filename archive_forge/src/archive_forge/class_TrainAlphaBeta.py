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
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class TrainAlphaBeta(Train):
    """Submit an AI Platform training job."""
    _SUPPORT_TPU_TF_VERSION = True

    @classmethod
    def Args(cls, parser):
        _AddSubmitTrainingArgs(parser)
        flags.AddKmsKeyFlag(parser, 'job')
        flags.NETWORK.AddToParser(parser)
        flags.AddCustomContainerFlags(parser, support_tpu_tf_version=cls._SUPPORT_TPU_TF_VERSION)
        parser.display_info.AddFormat(jobs_util.JOB_FORMAT)