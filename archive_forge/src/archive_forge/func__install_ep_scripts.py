from distutils import log
import distutils.command.install_scripts as orig
import os
import sys
from .._path import ensure_directory
def _install_ep_scripts(self):
    from pkg_resources import Distribution, PathMetadata
    from . import easy_install as ei
    ei_cmd = self.get_finalized_command('egg_info')
    dist = Distribution(ei_cmd.egg_base, PathMetadata(ei_cmd.egg_base, ei_cmd.egg_info), ei_cmd.egg_name, ei_cmd.egg_version)
    bs_cmd = self.get_finalized_command('build_scripts')
    exec_param = getattr(bs_cmd, 'executable', None)
    writer = ei.ScriptWriter
    if exec_param == sys.executable:
        exec_param = [exec_param]
    writer = writer.best()
    cmd = writer.command_spec_class.best().from_param(exec_param)
    for args in writer.get_args(dist, cmd.as_header()):
        self.write_script(*args)