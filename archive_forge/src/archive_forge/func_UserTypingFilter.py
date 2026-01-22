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
@filters.Condition
def UserTypingFilter(cli):
    """Determine if the input field is empty."""
    return cli.current_buffer.document.text and cli.current_buffer.document.text != cli.config.context