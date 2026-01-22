import sys
import os
class DistutilsMetaFinder:

    def find_spec(self, fullname, path, target=None):
        if path is not None and (not fullname.startswith('test.')):
            return None
        method_name = 'spec_for_{fullname}'.format(**locals())
        method = getattr(self, method_name, lambda: None)
        return method()

    def spec_for_distutils(self):
        if self.is_cpython():
            return None
        import importlib
        import importlib.abc
        import importlib.util
        try:
            mod = importlib.import_module('setuptools._distutils')
        except Exception:
            return None

        class DistutilsLoader(importlib.abc.Loader):

            def create_module(self, spec):
                mod.__name__ = 'distutils'
                return mod

            def exec_module(self, module):
                pass
        return importlib.util.spec_from_loader('distutils', DistutilsLoader(), origin=mod.__file__)

    @staticmethod
    def is_cpython():
        """
        Suppress supplying distutils for CPython (build and tests).
        Ref #2965 and #3007.
        """
        return os.path.isfile('pybuilddir.txt')

    def spec_for_pip(self):
        """
        Ensure stdlib distutils when running under pip.
        See pypa/pip#8761 for rationale.
        """
        if sys.version_info >= (3, 12) or self.pip_imported_during_build():
            return
        clear_distutils()
        self.spec_for_distutils = lambda: None

    @classmethod
    def pip_imported_during_build(cls):
        """
        Detect if pip is being imported in a build script. Ref #2355.
        """
        import traceback
        return any((cls.frame_file_is_setup(frame) for frame, line in traceback.walk_stack(None)))

    @staticmethod
    def frame_file_is_setup(frame):
        """
        Return True if the indicated frame suggests a setup.py file.
        """
        return frame.f_globals.get('__file__', '').endswith('setup.py')

    def spec_for_sensitive_tests(self):
        """
        Ensure stdlib distutils when running select tests under CPython.

        python/cpython#91169
        """
        clear_distutils()
        self.spec_for_distutils = lambda: None
    sensitive_tests = ['test.test_distutils', 'test.test_peg_generator', 'test.test_importlib'] if sys.version_info < (3, 10) else ['test.test_distutils']