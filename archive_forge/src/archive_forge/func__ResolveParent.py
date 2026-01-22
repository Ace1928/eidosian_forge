import gyp.common
import json
import os
import posixpath
def _ResolveParent(path, base_path_components):
    """Resolves |path|, which starts with at least one '../'. Returns an empty
  string if the path shouldn't be considered. See _AddSources() for a
  description of |base_path_components|."""
    depth = 0
    while path.startswith('../'):
        depth += 1
        path = path[3:]
    if depth > len(base_path_components):
        return ''
    if depth == len(base_path_components):
        return path
    return '/'.join(base_path_components[0:len(base_path_components) - depth]) + '/' + path