from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.interactive import lexer
import six
class ArgTokenType(enum.Enum):
    UNKNOWN = 0
    PREFIX = 1
    GROUP = 2
    COMMAND = 3
    FLAG = 4
    FLAG_ARG = 5
    POSITIONAL = 6
    SPECIAL = 7