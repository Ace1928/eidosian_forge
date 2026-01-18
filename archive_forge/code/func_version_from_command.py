import os
import subprocess as sp
import shlex
import simplejson as json
from traits.trait_errors import TraitError
from ... import config, logging, LooseVersion
from ...utils.provenance import write_provenance
from ...utils.misc import str2bool
from ...utils.filemanip import (
from ...utils.subprocess import run_command
from ...external.due import due
from .traits_extension import traits, isdefined, Undefined
from .specs import (
from .support import (
def version_from_command(self, flag='-v', cmd=None):
    iflogger.warning('version_from_command member of CommandLine was Deprecated in nipype-1.0.0 and deleted in 1.1.0')
    if cmd is None:
        cmd = self.cmd.split()[0]
    env = dict(os.environ)
    if which(cmd, env=env):
        out_environ = self._get_environ()
        env.update(out_environ)
        proc = sp.Popen(' '.join((cmd, flag)), shell=True, env=canonicalize_env(env), stdout=sp.PIPE, stderr=sp.PIPE)
        o, e = proc.communicate()
        return o