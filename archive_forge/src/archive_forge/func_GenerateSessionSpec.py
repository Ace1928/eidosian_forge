from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateSessionSpec(args):
    """Generate SessionSpec From Arguments."""
    module = dataplex_api.GetMessageModule()
    session_spec = module.GoogleCloudDataplexV1EnvironmentSessionSpec(enableFastStartup=args.session_enable_fast_startup, maxIdleDuration=args.session_max_idle_duration)
    return session_spec