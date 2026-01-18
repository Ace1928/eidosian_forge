from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
A resource which generates a random string.

    This is useful for configuring passwords and secrets on services. Random
    string can be generated from specified character sequences, which means
    that all characters will be randomly chosen from specified sequences, or
    with some classes, e.g. letterdigits, which means that all character will
    be randomly chosen from union of ascii letters and digits. Output string
    will be randomly generated string with specified length (or with length of
    32, if length property doesn't specified).
    