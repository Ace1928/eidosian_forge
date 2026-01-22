import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
class AnalysisLevel(enum.IntEnum):
    NONE = 0
    ACTIVITY = 1
    DEFINEDNESS = 2
    LIVENESS = 3