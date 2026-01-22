import _imp
import _io
import sys
import _warnings
import marshal
def _make_relax_case():
    if sys.platform.startswith(_CASE_INSENSITIVE_PLATFORMS):
        if sys.platform.startswith(_CASE_INSENSITIVE_PLATFORMS_STR_KEY):
            key = 'PYTHONCASEOK'
        else:
            key = b'PYTHONCASEOK'

        def _relax_case():
            """True if filenames must be checked case-insensitively and ignore environment flags are not set."""
            return not sys.flags.ignore_environment and key in _os.environ
    else:

        def _relax_case():
            """True if filenames must be checked case-insensitively."""
            return False
    return _relax_case