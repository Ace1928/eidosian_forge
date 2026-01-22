from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from .tab_complete import CompleterType
@staticmethod
def MakeOneOrMoreBindingsArgument():
    """Constructs an argument that takes multiple bindings."""
    return CommandArgument('binding', nargs='+', completer=CompleterType.NO_OP)