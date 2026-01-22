from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.help_provider import HelpProvider
class CommandOptions(HelpProvider):
    """Additional help about gsutil command-level options."""
    help_spec = HelpProvider.HelpSpec(help_name='options', help_name_aliases=['arg', 'args', 'cli', 'opt', 'opts'], help_type='additional_help', help_one_line_summary='Global Command Line Options', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})