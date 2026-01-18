import shutil
import os
import tempfile
import time
import Cython.Build.Dependencies
import Cython.Utils
from Cython.TestUtils import CythonTest
def test_recythonize_pyx_on_pxd_change(self):
    src_dir = tempfile.mkdtemp(prefix='src', dir=self.temp_dir)
    a_pxd = os.path.join(src_dir, 'a.pxd')
    a_pyx = os.path.join(src_dir, 'a.pyx')
    a_c = os.path.join(src_dir, 'a.c')
    dep_tree = Cython.Build.Dependencies.create_dependency_tree()
    with open(a_pxd, 'w') as f:
        f.write('cdef int value\n')
    with open(a_pyx, 'w') as f:
        f.write('value = 1\n')
    self.assertEqual({a_pxd, a_pyx}, dep_tree.all_dependencies(a_pyx))
    fresh_cythonize(a_pyx)
    time.sleep(1)
    with open(a_c) as f:
        a_c_contents1 = f.read()
    with open(a_pxd, 'w') as f:
        f.write('cdef double value\n')
    fresh_cythonize(a_pyx)
    with open(a_c) as f:
        a_c_contents2 = f.read()
    self.assertTrue('__pyx_v_1a_value = 1;' in a_c_contents1)
    self.assertFalse('__pyx_v_1a_value = 1;' in a_c_contents2)
    self.assertTrue('__pyx_v_1a_value = 1.0;' in a_c_contents2)
    self.assertFalse('__pyx_v_1a_value = 1.0;' in a_c_contents1)