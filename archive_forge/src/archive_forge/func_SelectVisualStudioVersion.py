import errno
import os
import re
import subprocess
import sys
import glob
def SelectVisualStudioVersion(version='auto', allow_fallback=True):
    """Select which version of Visual Studio projects to generate.

  Arguments:
    version: Hook to allow caller to force a particular version (vs auto).
  Returns:
    An object representing a visual studio project format version.
  """
    if version == 'auto':
        version = os.environ.get('GYP_MSVS_VERSION', 'auto')
    version_map = {'auto': ('16.0', '15.0', '14.0', '12.0', '10.0', '9.0', '8.0', '11.0'), '2005': ('8.0',), '2005e': ('8.0',), '2008': ('9.0',), '2008e': ('9.0',), '2010': ('10.0',), '2010e': ('10.0',), '2012': ('11.0',), '2012e': ('11.0',), '2013': ('12.0',), '2013e': ('12.0',), '2015': ('14.0',), '2017': ('15.0',), '2019': ('16.0',)}
    override_path = os.environ.get('GYP_MSVS_OVERRIDE_PATH')
    if override_path:
        msvs_version = os.environ.get('GYP_MSVS_VERSION')
        if not msvs_version:
            raise ValueError('GYP_MSVS_OVERRIDE_PATH requires GYP_MSVS_VERSION to be set to a particular version (e.g. 2010e).')
        return _CreateVersion(msvs_version, override_path, sdk_based=True)
    version = str(version)
    versions = _DetectVisualStudioVersions(version_map[version], 'e' in version)
    if not versions:
        if not allow_fallback:
            raise ValueError('Could not locate Visual Studio installation.')
        if version == 'auto':
            return _CreateVersion('2005', None)
        else:
            return _CreateVersion(version, None)
    return versions[0]