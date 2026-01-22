from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
class ParserData(object):
    """Parser data for the entire command.

    Attributes:
      allow_positional: bool, Allow positional arguments if True.
      ancestor_flag_args: [argparse.Action], The flags for all ancestor groups
        in the cli tree.
      cli_generator: cli.CLILoader, The builder used to generate this CLI.
      command_name: [str], The parts of the command name path.
      concept_handler: calliope.concepts.handlers.RuntimeHandler, a handler
        for concept args.
      defaults: {dest: default}, For all registered arguments.
      dests: [str], A list of the dests for all arguments.
      display_info: [display_info.DisplayInfo], The command display info object.
      flag_args: [ArgumentInterceptor], The flag arguments.
      positional_args: [ArgumentInterceptor], The positional args.
      positional_completers: {Completer}, The set of completers for positionals.
      required: [str], The dests for all required arguments.
    """

    def __init__(self, command_name, cli_generator, allow_positional):
        self.command_name = command_name
        self.cli_generator = cli_generator
        self.allow_positional = allow_positional
        self.ancestor_flag_args = []
        self.concept_handler = None
        self.concepts = None
        self.defaults = {}
        self.dests = []
        self.display_info = display_info.DisplayInfo()
        self.flag_args = []
        self.positional_args = []
        self.positional_completers = set()
        self.required = []