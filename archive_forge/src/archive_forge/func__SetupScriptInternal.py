import errno
import os
import re
import subprocess
import sys
import glob
def _SetupScriptInternal(self, target_arch):
    """Returns a command (with arguments) to be used to set up the
    environment."""
    assert target_arch in ('x86', 'x64'), 'target_arch not supported'
    sdk_dir = os.environ.get('WindowsSDKDir', '')
    setup_path = JoinPath(sdk_dir, 'Bin', 'SetEnv.Cmd')
    if self.sdk_based and sdk_dir and os.path.exists(setup_path):
        return [setup_path, '/' + target_arch]
    is_host_arch_x64 = os.environ.get('PROCESSOR_ARCHITECTURE') == 'AMD64' or os.environ.get('PROCESSOR_ARCHITEW6432') == 'AMD64'
    if self.short_name >= '2017':
        script_path = JoinPath(self.path, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat')
        host_arch = 'amd64' if is_host_arch_x64 else 'x86'
        msvc_target_arch = 'amd64' if target_arch == 'x64' else 'x86'
        arg = host_arch
        if host_arch != msvc_target_arch:
            arg += '_' + msvc_target_arch
        return [script_path, arg]
    vcvarsall = JoinPath(self.path, 'VC', 'vcvarsall.bat')
    if target_arch == 'x86':
        if self.short_name >= '2013' and self.short_name[-1] != 'e' and is_host_arch_x64:
            return [vcvarsall, 'amd64_x86']
        else:
            return [JoinPath(self.path, 'Common7', 'Tools', 'vsvars32.bat')]
    elif target_arch == 'x64':
        arg = 'x86_amd64'
        if self.short_name[-1] != 'e' and is_host_arch_x64:
            arg = 'amd64'
        return [vcvarsall, arg]