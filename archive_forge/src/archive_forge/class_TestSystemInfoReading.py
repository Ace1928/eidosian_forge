import os
import shutil
import pytest
from tempfile import mkstemp, mkdtemp
from subprocess import Popen, PIPE
import importlib.metadata
from distutils.errors import DistutilsError
from numpy.testing import assert_, assert_equal, assert_raises
from numpy.distutils import ccompiler, customized_ccompiler
from numpy.distutils.system_info import system_info, ConfigParser, mkl_info
from numpy.distutils.system_info import AliasedOptionError
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs
from numpy.distutils import _shell_utils
class TestSystemInfoReading:

    def setup_method(self):
        """ Create the libraries """
        self._dir1 = mkdtemp()
        self._src1 = os.path.join(self._dir1, 'foo.c')
        self._lib1 = os.path.join(self._dir1, 'libfoo.so')
        self._dir2 = mkdtemp()
        self._src2 = os.path.join(self._dir2, 'bar.c')
        self._lib2 = os.path.join(self._dir2, 'libbar.so')
        global simple_site, site_cfg
        site_cfg = simple_site.format(**{'dir1': self._dir1, 'lib1': self._lib1, 'dir2': self._dir2, 'lib2': self._lib2, 'pathsep': os.pathsep, 'lib2_escaped': _shell_utils.NativeParser.join([self._lib2])})
        fd, self._sitecfg = mkstemp()
        os.close(fd)
        with open(self._sitecfg, 'w') as fd:
            fd.write(site_cfg)
        with open(self._src1, 'w') as fd:
            fd.write(fakelib_c_text)
        with open(self._src2, 'w') as fd:
            fd.write(fakelib_c_text)

        def site_and_parse(c, site_cfg):
            c.files = [site_cfg]
            c.parse_config_files()
            return c
        self.c_default = site_and_parse(get_class('default'), self._sitecfg)
        self.c_temp1 = site_and_parse(get_class('temp1'), self._sitecfg)
        self.c_temp2 = site_and_parse(get_class('temp2'), self._sitecfg)
        self.c_dup_options = site_and_parse(get_class('duplicate_options'), self._sitecfg)

    def teardown_method(self):
        try:
            shutil.rmtree(self._dir1)
        except Exception:
            pass
        try:
            shutil.rmtree(self._dir2)
        except Exception:
            pass
        try:
            os.remove(self._sitecfg)
        except Exception:
            pass

    def test_all(self):
        tsi = self.c_default
        assert_equal(tsi.get_lib_dirs(), [self._dir1, self._dir2])
        assert_equal(tsi.get_libraries(), [self._lib1, self._lib2])
        assert_equal(tsi.get_runtime_lib_dirs(), [self._dir1])
        extra = tsi.calc_extra_info()
        assert_equal(extra['extra_compile_args'], ['-I/fake/directory', '-I/path with/spaces', '-Os'])

    def test_temp1(self):
        tsi = self.c_temp1
        assert_equal(tsi.get_lib_dirs(), [self._dir1])
        assert_equal(tsi.get_libraries(), [self._lib1])
        assert_equal(tsi.get_runtime_lib_dirs(), [self._dir1])

    def test_temp2(self):
        tsi = self.c_temp2
        assert_equal(tsi.get_lib_dirs(), [self._dir2])
        assert_equal(tsi.get_libraries(), [self._lib2])
        assert_equal(tsi.get_runtime_lib_dirs(key='rpath'), [self._dir2])
        extra = tsi.calc_extra_info()
        assert_equal(extra['extra_link_args'], ['-Wl,-rpath=' + self._lib2])

    def test_duplicate_options(self):
        tsi = self.c_dup_options
        assert_raises(AliasedOptionError, tsi.get_option_single, 'mylib_libs', 'libraries')
        assert_equal(tsi.get_libs('mylib_libs', [self._lib1]), [self._lib1])
        assert_equal(tsi.get_libs('libraries', [self._lib2]), [self._lib2])

    @pytest.mark.skipif(not HAVE_COMPILER, reason='Missing compiler')
    def test_compile1(self):
        c = customized_ccompiler()
        previousDir = os.getcwd()
        try:
            os.chdir(self._dir1)
            c.compile([os.path.basename(self._src1)], output_dir=self._dir1)
            assert_(os.path.isfile(self._src1.replace('.c', '.o')) or os.path.isfile(self._src1.replace('.c', '.obj')))
        finally:
            os.chdir(previousDir)

    @pytest.mark.skipif(not HAVE_COMPILER, reason='Missing compiler')
    @pytest.mark.skipif('msvc' in repr(ccompiler.new_compiler()), reason='Fails with MSVC compiler ')
    def test_compile2(self):
        tsi = self.c_temp2
        c = customized_ccompiler()
        extra_link_args = tsi.calc_extra_info()['extra_link_args']
        previousDir = os.getcwd()
        try:
            os.chdir(self._dir2)
            c.compile([os.path.basename(self._src2)], output_dir=self._dir2, extra_postargs=extra_link_args)
            assert_(os.path.isfile(self._src2.replace('.c', '.o')))
        finally:
            os.chdir(previousDir)
    HAS_MKL = 'mkl_rt' in mkl_info().calc_libraries_info().get('libraries', [])

    @pytest.mark.xfail(HAS_MKL, reason="`[DEFAULT]` override doesn't work if numpy is built with MKL support")
    def test_overrides(self):
        previousDir = os.getcwd()
        cfg = os.path.join(self._dir1, 'site.cfg')
        shutil.copy(self._sitecfg, cfg)
        try:
            os.chdir(self._dir1)
            info = mkl_info()
            lib_dirs = info.cp['ALL']['library_dirs'].split(os.pathsep)
            assert info.get_lib_dirs() != lib_dirs
            with open(cfg) as fid:
                mkl = fid.read().replace('[ALL]', '[mkl]', 1)
            with open(cfg, 'w') as fid:
                fid.write(mkl)
            info = mkl_info()
            assert info.get_lib_dirs() == lib_dirs
            with open(cfg) as fid:
                dflt = fid.read().replace('[mkl]', '[DEFAULT]', 1)
            with open(cfg, 'w') as fid:
                fid.write(dflt)
            info = mkl_info()
            assert info.get_lib_dirs() == lib_dirs
        finally:
            os.chdir(previousDir)