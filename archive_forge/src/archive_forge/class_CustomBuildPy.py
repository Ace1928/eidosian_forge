import os
from setuptools import find_packages
from setuptools import setup
from setuptools.command import build_py
from setuptools.command import sdist
class CustomBuildPy(build_py.build_py):
    """Excludes update command from package-installed versions of gsutil."""

    def byte_compile(self, files):
        for filename in files:
            if 'gslib/commands/update.py' in filename:
                os.unlink(filename)
        build_py.build_py.byte_compile(self, files)

    def run(self):
        if not self.dry_run:
            PlaceNeededFiles(self, self.build_lib)
            build_py.build_py.run(self)