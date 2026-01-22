import os
from setuptools import find_packages
from setuptools import setup
from setuptools.command import build_py
from setuptools.command import sdist
def PlaceNeededFiles(self, target_dir):
    """Populates necessary files into the gslib module and unit test modules."""
    target_dir = os.path.join(target_dir, 'gslib')
    self.mkpath(target_dir)
    with open(os.path.join(target_dir, 'VERSION'), 'w') as fp:
        fp.write(VERSION)
    with open(os.path.join(target_dir, 'CHECKSUM'), 'w') as fp:
        fp.write(CHECKSUM)