from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import flags
from googlecloudsdk.command_lib.spanner import migration_backend
class Web(base.BinaryBackedCommand):
    """Run the web UI assistant for schema migrations."""
    detailed_help = {'EXAMPLES': textwrap.dedent('        To run the web UI assistant:\n\n          $ {command}\n      ')}

    @staticmethod
    def Args(parser):
        """Register the flags for this command."""
        flags.GetSpannerMigrationWebPortFlag().AddToParser(parser)
        flags.GetSpannerMigrationWebOpenFlag().AddToParser(parser)

    def Run(self, args):
        """Run the web UI assistant."""
        command_executor = migration_backend.SpannerMigrationWrapper()
        env_vars = migration_backend.GetEnvArgsForCommand(extra_vars={'GCLOUD_HB_PLUGIN': 'true'})
        response = command_executor(command='web', open_flag=args.open, port=args.port, env=env_vars)
        self.exit_code = response.exit_code
        return self._DefaultOperationResponseHandler(response)