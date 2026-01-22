import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def WriteNewNinjaRule(self, name, args, description, win_shell_flags, env, pool, depfile=None):
    """Write out a new ninja "rule" statement for a given command.

        Returns the name of the new rule, and a copy of |args| with variables
        expanded."""
    if self.flavor == 'win':
        args = [self.msvs_settings.ConvertVSMacros(arg, self.base_to_build, config=self.config_name) for arg in args]
        description = self.msvs_settings.ConvertVSMacros(description, config=self.config_name)
    elif self.flavor == 'mac':
        args = [gyp.xcode_emulation.ExpandEnvVars(arg, env) for arg in args]
        description = gyp.xcode_emulation.ExpandEnvVars(description, env)
    rule_name = self.name
    if self.toolset == 'target':
        rule_name += '.' + self.toolset
    rule_name += '.' + name
    rule_name = re.sub('[^a-zA-Z0-9_]', '_', rule_name)
    protect = ['${root}', '${dirname}', '${source}', '${ext}', '${name}']
    protect = '(?!' + '|'.join(map(re.escape, protect)) + ')'
    description = re.sub(protect + '\\$', '_', description)
    rspfile = None
    rspfile_content = None
    args = [self.ExpandSpecial(arg, self.base_to_build) for arg in args]
    if self.flavor == 'win':
        rspfile = rule_name + '.$unique_name.rsp'
        run_in = '' if win_shell_flags.cygwin else ' ' + self.build_to_base
        if win_shell_flags.cygwin:
            rspfile_content = self.msvs_settings.BuildCygwinBashCommandLine(args, self.build_to_base)
        else:
            rspfile_content = gyp.msvs_emulation.EncodeRspFileList(args, win_shell_flags.quote)
        command = '%s gyp-win-tool action-wrapper $arch ' % sys.executable + rspfile + run_in
    else:
        env = self.ComputeExportEnvString(env)
        command = gyp.common.EncodePOSIXShellList(args)
        command = 'cd %s; ' % self.build_to_base + env + command
    self.ninja.rule(rule_name, command, description, depfile=depfile, restat=True, pool=pool, rspfile=rspfile, rspfile_content=rspfile_content)
    self.ninja.newline()
    return (rule_name, args)