from __future__ import unicode_literals
from distutils.command import install as du_install
from distutils import log
import email
import email.errors
import os
import re
import sys
import warnings
import pkg_resources
import setuptools
from setuptools.command import develop
from setuptools.command import easy_install
from setuptools.command import egg_info
from setuptools.command import install
from setuptools.command import install_scripts
from setuptools.command import sdist
from pbr import extra_files
from pbr import git
from pbr import options
import pbr.pbr_json
from pbr import testr_command
from pbr import version
import threading
from %(module_name)s import %(import_target)s
import sys
from %(module_name)s import %(import_target)s
def _get_version_from_pkg_metadata(package_name):
    """Get the version from package metadata if present.

    This looks for PKG-INFO if present (for sdists), and if not looks
    for METADATA (for wheels) and failing that will return None.
    """
    pkg_metadata_filenames = ['PKG-INFO', 'METADATA']
    pkg_metadata = {}
    for filename in pkg_metadata_filenames:
        try:
            with open(filename, 'r') as pkg_metadata_file:
                pkg_metadata = email.message_from_file(pkg_metadata_file)
        except (IOError, OSError, email.errors.MessageError):
            continue
    if pkg_metadata.get('Name', None) != package_name:
        return None
    return pkg_metadata.get('Version', None)