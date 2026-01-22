import os
import shutil
import tempfile
import fixtures
class NestedTempfile(fixtures.Fixture):
    """Nest all temporary files and directories inside another directory.

    This temporarily monkey-patches the default location that the `tempfile`
    package creates temporary files and directories in to be a new temporary
    directory. This new temporary directory is removed when the fixture is torn
    down.
    """

    def _setUp(self):
        tempdir = self.useFixture(TempDir()).path
        patch = fixtures.MonkeyPatch('tempfile.tempdir', tempdir)
        self.useFixture(patch)