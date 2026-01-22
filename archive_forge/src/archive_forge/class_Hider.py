import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
class Hider:
    hidden = True