import sys, os
import textwrap
class IndentedHelpFormatter(HelpFormatter):
    """Format help with indented section bodies.
    """

    def __init__(self, indent_increment=2, max_help_position=24, width=None, short_first=1):
        HelpFormatter.__init__(self, indent_increment, max_help_position, width, short_first)

    def format_usage(self, usage):
        return _('Usage: %s\n') % usage

    def format_heading(self, heading):
        return '%*s%s:\n' % (self.current_indent, '', heading)