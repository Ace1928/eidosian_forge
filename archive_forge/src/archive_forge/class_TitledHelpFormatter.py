import sys, os
import textwrap
class TitledHelpFormatter(HelpFormatter):
    """Format help with underlined section headers.
    """

    def __init__(self, indent_increment=0, max_help_position=24, width=None, short_first=0):
        HelpFormatter.__init__(self, indent_increment, max_help_position, width, short_first)

    def format_usage(self, usage):
        return '%s  %s\n' % (self.format_heading(_('Usage')), usage)

    def format_heading(self, heading):
        return '%s\n%s\n' % (heading, '=-'[self.level] * len(heading))