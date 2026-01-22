import argparse
import os
import sys
from argparse import ArgumentParser, HelpFormatter
from functools import partial
from io import TextIOBase
import django
from django.core import checks
from django.core.exceptions import ImproperlyConfigured
from django.core.management.color import color_style, no_style
from django.db import DEFAULT_DB_ALIAS, connections
class CommandParser(ArgumentParser):
    """
    Customized ArgumentParser class to improve some error messages and prevent
    SystemExit in several occasions, as SystemExit is unacceptable when a
    command is called programmatically.
    """

    def __init__(self, *, missing_args_message=None, called_from_command_line=None, **kwargs):
        self.missing_args_message = missing_args_message
        self.called_from_command_line = called_from_command_line
        super().__init__(**kwargs)

    def parse_args(self, args=None, namespace=None):
        if self.missing_args_message and (not (args or any((not arg.startswith('-') for arg in args)))):
            self.error(self.missing_args_message)
        return super().parse_args(args, namespace)

    def error(self, message):
        if self.called_from_command_line:
            super().error(message)
        else:
            raise CommandError('Error: %s' % message)

    def add_subparsers(self, **kwargs):
        parser_class = kwargs.get('parser_class', type(self))
        if issubclass(parser_class, CommandParser):
            kwargs['parser_class'] = partial(parser_class, called_from_command_line=self.called_from_command_line)
        return super().add_subparsers(**kwargs)