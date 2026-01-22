from winappdbg.textio import HexInput
from winappdbg.util import StaticClass, MemoryAddresses
from winappdbg import win32
import warnings
class HexPattern(RegExpPattern):
    """
    Hexadecimal pattern.

    Hex patterns must be in this form::
        "68 65 6c 6c 6f 20 77 6f 72 6c 64"  # "hello world"

    Spaces are optional. Capitalization of hex digits doesn't matter.
    This is exactly equivalent to the previous example::
        "68656C6C6F20776F726C64"            # "hello world"

    Wildcards are allowed, in the form of a C{?} sign in any hex digit::
        "5? 5? c3"          # pop register / pop register / ret
        "b8 ?? ?? ?? ??"    # mov eax, immediate value

    @type pattern: str
    @ivar pattern: Hexadecimal pattern.
    """

    def __new__(cls, pattern):
        """
        If the pattern is completely static (no wildcards are present) a
        L{BytePattern} is created instead. That's because searching for a
        fixed byte pattern is faster than searching for a regular expression.
        """
        if '?' not in pattern:
            return BytePattern(HexInput.hexadecimal(pattern))
        return object.__new__(cls, pattern)

    def __init__(self, hexa):
        """
        Hex patterns must be in this form::
            "68 65 6c 6c 6f 20 77 6f 72 6c 64"  # "hello world"

        Spaces are optional. Capitalization of hex digits doesn't matter.
        This is exactly equivalent to the previous example::
            "68656C6C6F20776F726C64"            # "hello world"

        Wildcards are allowed, in the form of a C{?} sign in any hex digit::
            "5? 5? c3"          # pop register / pop register / ret
            "b8 ?? ?? ?? ??"    # mov eax, immediate value

        @type  hexa: str
        @param hexa: Pattern to search for.
        """
        maxLength = len([x for x in hexa if x in '?0123456789ABCDEFabcdef']) / 2
        super(HexPattern, self).__init__(HexInput.pattern(hexa), maxLength=maxLength)