import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def _check_OSX_can_roundtrip(self, path, fs_enc=None):
    """Stop the test if it's about to fail or errors out.

        Until we get proper support on OSX for accented paths (in fact, any
        path whose NFD decomposition is different than the NFC one), this is
        the best way to keep test active (as opposed to disabling them
        completely). This is a stop gap. The tests should at least be rewritten
        so that the failing ones are clearly separated from the passing ones.
        """
    if fs_enc is None:
        fs_enc = sys.getfilesystemencoding()
    if sys.platform == 'darwin':
        encoded = path.encode(fs_enc)
        import unicodedata
        normal_thing = unicodedata.normalize('NFD', path)
        mac_encoded = normal_thing.encode(fs_enc)
        if mac_encoded != encoded:
            self.knownFailure('Unable to roundtrip path %r on OSX filesystem using encoding "%s"' % (path, fs_enc))