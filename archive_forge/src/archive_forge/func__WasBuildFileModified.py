import gyp.common
import json
import os
import posixpath
def _WasBuildFileModified(build_file, data, files, toplevel_dir):
    """Returns true if the build file |build_file| is either in |files| or
  one of the files included by |build_file| is in |files|. |toplevel_dir| is
  the root of the source tree."""
    if _ToLocalPath(toplevel_dir, _ToGypPath(build_file)) in files:
        if debug:
            print('gyp file modified', build_file)
        return True
    if len(data[build_file]['included_files']) <= 1:
        return False
    for include_file in data[build_file]['included_files'][1:]:
        rel_include_file = _ToGypPath(gyp.common.UnrelativePath(include_file, build_file))
        if _ToLocalPath(toplevel_dir, rel_include_file) in files:
            if debug:
                print('included gyp file modified, gyp_file=', build_file, 'included file=', rel_include_file)
            return True
    return False