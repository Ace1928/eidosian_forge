import os
from typing import List
from .. import forge as _mod_forge
from .. import registry, tests, urlutils
from ..forge import (Forge, MergeProposal, UnsupportedForge, determine_title,
class SampleForgeTestCase(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self._old_forges = _mod_forge.forges
        _mod_forge.forges = registry.Registry()
        self.forge = SampleForge()
        os.mkdir('hosted')
        SampleForge._add_location(urlutils.local_path_to_url(os.path.join(self.test_dir, 'hosted')))
        _mod_forge.forges.register('sample', self.forge)

    def tearDown(self):
        super().tearDown()
        _mod_forge.forges = self._old_forges
        SampleForge._locations = []