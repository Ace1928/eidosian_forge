import sys
import os
import multiprocessing as mp
import warnings
from numba.core.config import IS_WIN32, IS_OSX
from numba.core.errors import NumbaWarning
from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import (
from numba.cuda.cuda_paths import (
@skip_on_cudasim('Library detection unsupported in the simulator')
@unittest.skipUnless(has_mp_get_context, 'mp.get_context not available')
@skip_unless_conda_cudatoolkit('test assumes conda installed cudatoolkit')
class TestLibDeviceLookUp(LibraryLookupBase):

    def test_libdevice_path_decision(self):
        by, info, warns = self.remote_do(self.do_clear_envs)
        if has_cuda:
            self.assertEqual(by, 'Conda environment')
        else:
            self.assertEqual(by, '<unknown>')
            self.assertIsNone(info)
        self.assertFalse(warns)
        by, info, warns = self.remote_do(self.do_set_cuda_home)
        self.assertEqual(by, 'CUDA_HOME')
        self.assertEqual(info, os.path.join('mycudahome', 'nvvm', 'libdevice'))
        self.assertFalse(warns)
        if get_system_ctk() is None:
            by, info, warns = self.remote_do(self.do_clear_envs)
            self.assertEqual(by, '<unknown>')
            self.assertIsNone(info)
            self.assertFalse(warns)
        else:
            by, info, warns = self.remote_do(self.do_clear_envs)
            self.assertEqual(by, 'System')
            self.assertFalse(warns)

    @staticmethod
    def do_clear_envs():
        remove_env('CUDA_HOME')
        remove_env('CUDA_PATH')
        return (True, _get_libdevice_path_decision())

    @staticmethod
    def do_set_cuda_home():
        os.environ['CUDA_HOME'] = os.path.join('mycudahome')
        _fake_non_conda_env()
        return (True, _get_libdevice_path_decision())