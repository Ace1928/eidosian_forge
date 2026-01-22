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
class LocalInstallScripts(install_scripts.install_scripts):
    """Intercepts console scripts entry_points."""
    command_name = 'install_scripts'

    def _make_wsgi_scripts_only(self, dist, executable):
        try:
            header = easy_install.ScriptWriter.get_header('', executable)
        except AttributeError:
            header = easy_install.get_script_header('', executable)
        wsgi_script_template = ENTRY_POINTS_MAP['wsgi_scripts']
        for name, ep in dist.get_entry_map('wsgi_scripts').items():
            content = generate_script('wsgi_scripts', ep, header, wsgi_script_template)
            self.write_script(name, content)

    def run(self):
        import distutils.command.install_scripts
        self.run_command('egg_info')
        if self.distribution.scripts:
            distutils.command.install_scripts.install_scripts.run(self)
        else:
            self.outfiles = []
        ei_cmd = self.get_finalized_command('egg_info')
        dist = pkg_resources.Distribution(ei_cmd.egg_base, pkg_resources.PathMetadata(ei_cmd.egg_base, ei_cmd.egg_info), ei_cmd.egg_name, ei_cmd.egg_version)
        bs_cmd = self.get_finalized_command('build_scripts')
        executable = getattr(bs_cmd, 'executable', easy_install.sys_executable)
        if 'bdist_wheel' in self.distribution.have_run:
            self._make_wsgi_scripts_only(dist, executable)
        if self.no_ep:
            return
        if os.name != 'nt':
            get_script_args = override_get_script_args
        else:
            get_script_args = easy_install.get_script_args
            executable = '"%s"' % executable
        for args in get_script_args(dist, executable):
            self.write_script(*args)