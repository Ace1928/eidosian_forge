from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import six
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import cat_helper
from gslib.utils import constants
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
class CatCommand(Command):
    """Implementation of gsutil cat command."""
    command_spec = Command.CreateCommandSpec('cat', command_name_aliases=[], usage_synopsis=_SYNOPSIS, min_args=1, max_args=constants.NO_MAX, supported_sub_args='hr:', file_url_ok=False, provider_url_ok=False, urls_start_arg=0, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeZeroOrMoreCloudURLsArgument()])
    help_spec = Command.HelpSpec(help_name='cat', help_name_aliases=[], help_type='command_help', help_one_line_summary='Concatenate object content to stdout', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})
    gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'cat'], flag_map={'-h': GcloudStorageFlag('-d'), '-r': GcloudStorageFlag('-r')})

    def RunCommand(self):
        """Command entry point for the cat command."""
        show_header = False
        request_range = None
        start_byte = 0
        end_byte = None
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-h':
                    show_header = True
                elif o == '-r':
                    request_range = a.strip()
                    if request_range == '-':
                        continue
                    range_matcher = re.compile('^(?P<start>[0-9]+)-(?P<end>[0-9]*)$|^(?P<endslice>-[0-9]+)$')
                    range_match = range_matcher.match(request_range)
                    if not range_match:
                        raise CommandException('Invalid range (%s)' % request_range)
                    if range_match.group('start'):
                        start_byte = long(range_match.group('start'))
                    if range_match.group('end'):
                        end_byte = long(range_match.group('end'))
                    if range_match.group('endslice'):
                        start_byte = long(range_match.group('endslice'))
                else:
                    self.RaiseInvalidArgumentException()
        return cat_helper.CatHelper(self).CatUrlStrings(self.args, show_header=show_header, start_byte=start_byte, end_byte=end_byte)