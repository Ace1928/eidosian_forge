import os
import sys
import json
from os.path import isdir
import subprocess
import pathlib
import setuptools
from setuptools import Command
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from distutils.command.build import build
from setuptools.dist import Distribution
import shutil
class InstallWithMinecraftLib(install_lib):
    """Overrides the build command in install lib to build the minecraft library
    and place it in the build directory.
    """

    def build(self):
        super().build()