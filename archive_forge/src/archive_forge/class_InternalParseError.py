from typing import Dict, Type
from parso import tree
from parso.pgen2.generator import ReservedString
class InternalParseError(Exception):
    """
    Exception to signal the parser is stuck and error recovery didn't help.
    Basically this shouldn't happen. It's a sign that something is really
    wrong.
    """

    def __init__(self, msg, type_, value, start_pos):
        Exception.__init__(self, '%s: type=%r, value=%r, start_pos=%r' % (msg, type_.name, value, start_pos))
        self.msg = msg
        self.type = type
        self.value = value
        self.start_pos = start_pos