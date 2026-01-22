import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class InstrumentedDirState(dirstate.DirState):
    """An DirState with instrumented sha1 functionality."""

    def __init__(self, path, sha1_provider, worth_saving_limit=0, use_filesystem_for_exec=True):
        super().__init__(path, sha1_provider, worth_saving_limit=worth_saving_limit, use_filesystem_for_exec=use_filesystem_for_exec)
        self._time_offset = 0
        self._log = []
        self._sha1_provider = sha1_provider
        self._sha1_file = self._sha1_file_and_log

    def _sha_cutoff_time(self):
        timestamp = super()._sha_cutoff_time()
        self._cutoff_time = timestamp + self._time_offset

    def _sha1_file_and_log(self, abspath):
        self._log.append(('sha1', abspath))
        return self._sha1_provider.sha1(abspath)

    def _read_link(self, abspath, old_link):
        self._log.append(('read_link', abspath, old_link))
        return super()._read_link(abspath, old_link)

    def _lstat(self, abspath, entry):
        self._log.append(('lstat', abspath))
        return super()._lstat(abspath, entry)

    def _is_executable(self, mode, old_executable):
        self._log.append(('is_exec', mode, old_executable))
        return super()._is_executable(mode, old_executable)

    def adjust_time(self, secs):
        """Move the clock forward or back.

        :param secs: The amount to adjust the clock by. Positive values make it
        seem as if we are in the future, negative values make it seem like we
        are in the past.
        """
        self._time_offset += secs
        self._cutoff_time = None