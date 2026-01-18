from fontTools.misc.plistlib import dump, dumps, load, loads
from fontTools.misc.textTools import tobytes
from fontTools.ufoLib.utils import deprecated
@deprecated("Use 'fontTools.misc.plistlib.dumps' instead")
def writePlistToString(value):
    return dumps(value, use_builtin_types=False)