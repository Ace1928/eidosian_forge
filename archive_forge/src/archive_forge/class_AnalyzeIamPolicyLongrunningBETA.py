from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.asset import flags
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class AnalyzeIamPolicyLongrunningBETA(AnalyzeIamPolicyLongrunning):
    """Analyzes IAM policies that match a request asynchronously and writes the results to Google Cloud Storage or BigQuery destination."""

    @staticmethod
    def Args(parser):
        AnalyzeIamPolicyLongrunning.Args(parser)