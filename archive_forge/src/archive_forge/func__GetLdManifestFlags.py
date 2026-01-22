import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _GetLdManifestFlags(self, config, name, gyp_to_build_path, allow_isolation, build_dir):
    """Returns a 3-tuple:
        - the set of flags that need to be added to the link to generate
          a default manifest
        - the intermediate manifest that the linker will generate that should be
          used to assert it doesn't add anything to the merged one.
        - the list of all the manifest files to be merged by the manifest tool and
          included into the link."""
    generate_manifest = self._Setting(('VCLinkerTool', 'GenerateManifest'), config, default='true')
    if generate_manifest != 'true':
        return (['/MANIFEST:NO'], [], [])
    output_name = name + '.intermediate.manifest'
    flags = ['/MANIFEST', '/ManifestFile:' + output_name]
    flags.append('/MANIFESTUAC:NO')
    config = self._TargetConfig(config)
    enable_uac = self._Setting(('VCLinkerTool', 'EnableUAC'), config, default='true')
    manifest_files = []
    generated_manifest_outer = "<?xml version='1.0' encoding='UTF-8' standalone='yes'?><assembly xmlns='urn:schemas-microsoft-com:asm.v1' manifestVersion='1.0'>%s</assembly>"
    if enable_uac == 'true':
        execution_level = self._Setting(('VCLinkerTool', 'UACExecutionLevel'), config, default='0')
        execution_level_map = {'0': 'asInvoker', '1': 'highestAvailable', '2': 'requireAdministrator'}
        ui_access = self._Setting(('VCLinkerTool', 'UACUIAccess'), config, default='false')
        inner = '\n<trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n  <security>\n    <requestedPrivileges>\n      <requestedExecutionLevel level=\'{}\' uiAccess=\'{}\' />\n    </requestedPrivileges>\n  </security>\n</trustInfo>'.format(execution_level_map[execution_level], ui_access)
    else:
        inner = ''
    generated_manifest_contents = generated_manifest_outer % inner
    generated_name = name + '.generated.manifest'
    build_dir_generated_name = os.path.join(build_dir, generated_name)
    gyp.common.EnsureDirExists(build_dir_generated_name)
    f = gyp.common.WriteOnDiff(build_dir_generated_name)
    f.write(generated_manifest_contents)
    f.close()
    manifest_files = [generated_name]
    if allow_isolation:
        flags.append('/ALLOWISOLATION')
    manifest_files += self._GetAdditionalManifestFiles(config, gyp_to_build_path)
    return (flags, output_name, manifest_files)