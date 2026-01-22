from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.interactive import help_window
from prompt_toolkit import enums
from prompt_toolkit import filters
from prompt_toolkit import layout
from prompt_toolkit import shortcuts
from prompt_toolkit import token
from prompt_toolkit.layout import containers
from prompt_toolkit.layout import controls
from prompt_toolkit.layout import dimension
from prompt_toolkit.layout import margins
from prompt_toolkit.layout import menus
from prompt_toolkit.layout import processors
from prompt_toolkit.layout import prompt
from prompt_toolkit.layout import screen
from prompt_toolkit.layout import toolbars as pt_toolbars
def GetHeight(cli):
    """Determine the height for the input buffer."""
    if cli.config.completion_menu_lines and (not cli.is_done):
        buf = cli.current_buffer
        if UserTypingFilter(cli) or buf.complete_state:
            return dimension.LayoutDimension(min=cli.config.completion_menu_lines + 1)
    return dimension.LayoutDimension()