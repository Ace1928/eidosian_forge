from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import cp_command_util
from googlecloudsdk.command_lib.storage import flags
class Cp(base.Command):
    """Upload, download, and copy Cloud Storage objects."""
    detailed_help = {'DESCRIPTION': '\n      Copy data between your local file system and the cloud, within the cloud,\n      and between cloud storage providers.\n      ', 'EXAMPLES': "\n\n      The following command uploads all text files from the local directory to a\n      bucket:\n\n        $ {command} *.txt gs://my-bucket\n\n      The following command downloads all text files from a bucket to your\n      current directory:\n\n        $ {command} gs://my-bucket/*.txt .\n\n      The following command transfers all text files from a bucket to a\n      different cloud storage provider:\n\n        $ {command} gs://my-bucket/*.txt s3://my-bucket\n\n      Use the `--recursive` option to copy an entire directory tree. The\n      following command uploads the directory tree ``dir'':\n\n        $ {command} --recursive dir gs://my-bucket\n\n      Recursive listings are similar to adding `**` to a query, except\n      `**` matches only cloud objects and will not match prefixes. For\n      example, the following would not match ``gs://my-bucket/dir/log.txt''\n\n        $ {command} gs://my-bucket/**/dir dir\n\n      `**` retrieves a flat list of objects in a single API call. However, `**`\n      matches folders for non-cloud queries. For example, a folder ``dir''\n      would be copied in the following.\n\n        $ {command} ~/Downloads/**/dir gs://my-bucket\n      "}

    @classmethod
    def Args(cls, parser):
        cp_command_util.add_cp_and_mv_flags(parser)
        cp_command_util.add_recursion_flag(parser)
        flags.add_per_object_retention_flags(parser)

    def Run(self, args):
        self.exit_code = cp_command_util.run_cp(args)