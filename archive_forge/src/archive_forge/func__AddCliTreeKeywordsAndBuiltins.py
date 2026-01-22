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
def _AddCliTreeKeywordsAndBuiltins(root):
    """Adds keywords and builtins to the CLI tree root."""
    node = cli_tree.Node(command='exit', description='Exit the interactive shell.', positionals=[{'default': '0', 'description': 'The exit status.', 'name': 'status', 'nargs': '?', 'required': False, 'value': 'STATUS'}])
    node[parser.LOOKUP_IS_GROUP] = False
    root[parser.LOOKUP_COMMANDS]['exit'] = node
    for name in ['!', '{', 'do', 'elif', 'else', 'if', 'then', 'time', 'until', 'while']:
        node = cli_tree.Node(name)
        node[parser.LOOKUP_IS_GROUP] = False
        node[parser.LOOKUP_IS_SPECIAL] = True
        root[parser.LOOKUP_COMMANDS][name] = node
    for name in ['break', 'case', 'continue', 'done', 'esac', 'fi']:
        node = cli_tree.Node(name)
        node[parser.LOOKUP_IS_GROUP] = False
        root[parser.LOOKUP_COMMANDS][name] = node