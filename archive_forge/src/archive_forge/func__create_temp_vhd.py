import os
import tempfile
from os_win import constants
from os_win.tests.functional import test_base
from os_win import utilsfactory
def _create_temp_vhd(self, size_mb=32, vhd_type=constants.VHD_TYPE_DYNAMIC):
    f = tempfile.TemporaryFile(suffix='.vhdx', prefix='oswin_vhdtest_')
    f.close()
    self._vhdutils.create_vhd(f.name, vhd_type, max_internal_size=size_mb << 20)
    self.addCleanup(os.unlink, f.name)
    return f.name