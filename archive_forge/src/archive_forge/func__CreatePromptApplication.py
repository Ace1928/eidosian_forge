from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.interactive import bindings
from googlecloudsdk.command_lib.interactive import bindings_vi
from googlecloudsdk.command_lib.interactive import completer
from googlecloudsdk.command_lib.interactive import coshell as interactive_coshell
from googlecloudsdk.command_lib.interactive import debug as interactive_debug
from googlecloudsdk.command_lib.interactive import layout
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.command_lib.interactive import style as interactive_style
from googlecloudsdk.command_lib.meta import generate_cli_trees
from googlecloudsdk.core import config as core_config
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from prompt_toolkit import application as pt_application
from prompt_toolkit import auto_suggest
from prompt_toolkit import buffer as pt_buffer
from prompt_toolkit import document
from prompt_toolkit import enums
from prompt_toolkit import filters
from prompt_toolkit import history as pt_history
from prompt_toolkit import interface
from prompt_toolkit import shortcuts
from prompt_toolkit import token
from prompt_toolkit.layout import processors as pt_layout
def _CreatePromptApplication(self, config, multiline):
    """Creates a shell prompt Application."""
    return pt_application.Application(layout=layout.CreatePromptLayout(config=config, extra_input_processors=[Context()], get_bottom_status_tokens=self._GetBottomStatusTokens, get_bottom_toolbar_tokens=self._GetBottomToolbarTokens, get_continuation_tokens=None, get_debug_tokens=self._GetDebugTokens, get_prompt_tokens=None, is_password=False, lexer=None, multiline=filters.Condition(lambda cli: multiline()), show_help=filters.Condition(lambda _: self.key_bindings.help_key.toggle), wrap_lines=True), buffer=self.default_buffer, clipboard=None, erase_when_done=False, get_title=None, key_bindings_registry=self.key_bindings_registry, mouse_support=False, reverse_vi_search_direction=True, style=interactive_style.GetDocumentStyle())