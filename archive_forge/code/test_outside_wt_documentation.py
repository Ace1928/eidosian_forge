import os
import tempfile
from breezy import osutils, tests, transport, urlutils
Test that brz gives proper errors outside of a working tree.