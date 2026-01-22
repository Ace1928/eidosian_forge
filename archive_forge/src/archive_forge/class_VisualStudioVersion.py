import errno
import os
import re
import subprocess
import sys
import glob
class VisualStudioVersion:
    """Information regarding a version of Visual Studio."""

    def __init__(self, short_name, description, solution_version, project_version, flat_sln, uses_vcxproj, path, sdk_based, default_toolset=None, compatible_sdks=None):
        self.short_name = short_name
        self.description = description
        self.solution_version = solution_version
        self.project_version = project_version
        self.flat_sln = flat_sln
        self.uses_vcxproj = uses_vcxproj
        self.path = path
        self.sdk_based = sdk_based
        self.default_toolset = default_toolset
        compatible_sdks = compatible_sdks or []
        compatible_sdks.sort(key=lambda v: float(v.replace('v', '')), reverse=True)
        self.compatible_sdks = compatible_sdks

    def ShortName(self):
        return self.short_name

    def Description(self):
        """Get the full description of the version."""
        return self.description

    def SolutionVersion(self):
        """Get the version number of the sln files."""
        return self.solution_version

    def ProjectVersion(self):
        """Get the version number of the vcproj or vcxproj files."""
        return self.project_version

    def FlatSolution(self):
        return self.flat_sln

    def UsesVcxproj(self):
        """Returns true if this version uses a vcxproj file."""
        return self.uses_vcxproj

    def ProjectExtension(self):
        """Returns the file extension for the project."""
        return self.uses_vcxproj and '.vcxproj' or '.vcproj'

    def Path(self):
        """Returns the path to Visual Studio installation."""
        return self.path

    def ToolPath(self, tool):
        """Returns the path to a given compiler tool. """
        return os.path.normpath(os.path.join(self.path, 'VC/bin', tool))

    def DefaultToolset(self):
        """Returns the msbuild toolset version that will be used in the absence
    of a user override."""
        return self.default_toolset

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

    def SetupScript(self, target_arch):
        script_data = self._SetupScriptInternal(target_arch)
        script_path = script_data[0]
        if not os.path.exists(script_path):
            raise Exception('%s is missing - make sure VC++ tools are installed.' % script_path)
        return script_data