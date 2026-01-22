import abc
import argparse
import os
from zunclient.common.apiclient import exceptions
@classmethod
def add_opts(cls, parser):
    """Populate the parser with the options for this plugin."""
    for opt in cls.opt_names:
        if opt not in BaseAuthPlugin.common_opt_names:
            cls._parser_add_opt(parser, opt)