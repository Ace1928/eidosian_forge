import importlib.util
import sys
def _module_matches_namespace(self, fullname):
    """Figure out if the target module is vendored."""
    root, base, target = fullname.partition(self.root_name + '.')
    return not root and any(map(target.startswith, self.vendored_names))