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
def _get_revno_and_last_tag(git_dir):
    """Return the commit data about the most recent tag.

    We use git-describe to find this out, but if there are no
    tags then we fall back to counting commits since the beginning
    of time.
    """
    changelog = git._iter_log_oneline(git_dir=git_dir)
    row_count = 0
    for row_count, (ignored, tag_set, ignored) in enumerate(changelog):
        version_tags = set()
        semver_to_tag = dict()
        for tag in list(tag_set):
            try:
                semver = version.SemanticVersion.from_pip_string(tag)
                semver_to_tag[semver] = tag
                version_tags.add(semver)
            except Exception:
                pass
        if version_tags:
            return (semver_to_tag[max(version_tags)], row_count)
    return ('', row_count)