from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import sys
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import du_command_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import regex_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core.util import files
class Du(base.Command):
    """Displays the amount of space in bytes used by storage resources."""
    detailed_help = {'DESCRIPTION': '\n      Displays the amount of space in bytes used by the objects in a bucket,\n      subdirectory, or project. This command calculates the current space usage\n      by making a series of object listing requests, which can take a long time\n      for large buckets. If your bucket contains hundreds of thousands of\n      objects, or if you want to monitor your bucket size over time, use\n      Monitoring instead, as described in [Get bucket size](https://cloud.google.com/storage/docs/getting-bucket-size)\n      ', 'EXAMPLES': '\n\n      To list the size of each object in a bucket:\n\n        $ {command} gs://bucketname\n\n      To list the size of each object in the prefix subdirectory:\n\n        $ {command} gs://bucketname/prefix/*\n\n      To print the total number of bytes in a bucket in human-readable form:\n\n        $ {command} -ch gs://bucketname\n\n      To see a summary of the total number of bytes in two given buckets:\n\n        $ {command} -s gs://bucket1 gs://bucket2\n\n      To list the size of each object in a bucket with Object Versioning\n      enabled, including noncurrent objects:\n\n        $ {command} -a gs://bucketname\n\n      To list the size of each object in a bucket, except objects that end in\n      ".bak", with each object printed ending in a null byte:\n\n        $ {command} -e "*.bak" -0 gs://bucketname\n\n      To list the size of each bucket in a project and the total size of the\n      project:\n\n        $ {command} --summarize --readable-sizes --total\n      '}

    @staticmethod
    def Args(parser):
        parser.add_argument('url', nargs='*', help='The url of objects to list.')
        parser.add_argument('-0', '--zero-terminator', action='store_true', help='Ends each output line with a 0 byte rather than a newline. You can use this to make the output machine-readable.')
        parser.add_argument('-a', '--all-versions', action='store_true', help='Includes noncurrent object versions for a bucket with Object Versioning enabled. Also prints the generation and metageneration number for each listed object.')
        parser.add_argument('-c', '--total', action='store_true', help='Includes a total size of all input sources.')
        parser.add_argument('-e', '--exclude-name-pattern', action='append', default=[], help='Exclude a pattern from the report. Example: `-e "*.o"` excludes any object that ends in ".o". Can be specified multiple times.')
        parser.add_argument('-r', '--readable-sizes', action='store_true', help='Prints object sizes in human-readable format. For example, 1 KiB, 234 MiB, or 2GiB.')
        parser.add_argument('-s', '--summarize', action='store_true', help='Displays only the summary for each argument.')
        parser.add_argument('-X', '--exclude-name-pattern-file', help='Similar to -e, but excludes patterns from the given file. The patterns to exclude should be listed one per line.')
        flags.add_additional_headers_flag(parser)

    def Run(self, args):
        use_gsutil_style = flags.check_if_use_gsutil_style(args)
        if args.url:
            storage_urls = []
            for url_string in args.url:
                url_object = storage_url.storage_url_from_string(url_string)
                if not isinstance(url_object, storage_url.CloudUrl):
                    raise errors.InvalidUrlError('Du only works for valid cloud URLs. {} is an invalid cloud URL.'.format(url_object.url_string))
                storage_urls.append(url_object)
        else:
            storage_urls = [storage_url.CloudUrl(cloud_api.DEFAULT_PROVIDER)]
        exclude_fnmatch_strings = args.exclude_name_pattern
        if args.exclude_name_pattern_file:
            if args.exclude_name_pattern_file == '-':
                exclude_fnmatch_strings.extend([line.strip() for line in sys.stdin])
            else:
                with files.FileReader(args.exclude_name_pattern_file) as file:
                    exclude_fnmatch_strings.extend([line.strip() for line in file])
        exclude_regex_strings = [fnmatch.translate(pattern) for pattern in exclude_fnmatch_strings]
        du_command_util.DuExecutor(cloud_urls=storage_urls, exclude_patterns=regex_util.Patterns(exclude_regex_strings), object_state=flags.get_object_state_from_flags(args), readable_sizes=args.readable_sizes, summarize=args.summarize, total=args.total, use_gsutil_style=use_gsutil_style, zero_terminator=args.zero_terminator).list_urls()