import sys as _sys
from absl.app import run as _run
from tensorflow.python.platform import flags
from tensorflow.python.util.tf_export import tf_export
def _parse_flags_tolerate_undef(argv):
    """Parse args, returning any unknown flags (ABSL defaults to crashing)."""
    return flags.FLAGS(_sys.argv if argv is None else argv, known_only=True)