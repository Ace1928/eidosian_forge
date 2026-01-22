import os
from typing import List
from .. import forge as _mod_forge
from .. import registry, tests, urlutils
from ..forge import (Forge, MergeProposal, UnsupportedForge, determine_title,
class DetermineTitleTests(tests.TestCase):

    def test_determine_title(self):
        self.assertEqual('Make some change', determine_title('Make some change.\n\nAnd here are some more details.\n'))
        self.assertEqual('Make some change', determine_title('Make some change. And another one.\n\nWith details.\n'))
        self.assertEqual('Release version 5.1', determine_title('Release version 5.1\n\nAnd here are some more details.\n'))
        self.assertEqual('Release version 5.1', determine_title('\nRelease version 5.1\n\nAnd here are some more details.\n'))