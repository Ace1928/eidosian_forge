import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def ComputeDeps(self, spec):
    """Compute the dependencies of a gyp spec.

        Returns a tuple (deps, link_deps), where each is a list of
        filenames that will need to be put in front of make for either
        building (deps) or linking (link_deps).
        """
    deps = []
    link_deps = []
    if 'dependencies' in spec:
        deps.extend([target_outputs[dep] for dep in spec['dependencies'] if target_outputs[dep]])
        for dep in spec['dependencies']:
            if dep in target_link_deps:
                link_deps.append(target_link_deps[dep])
        deps.extend(link_deps)
    return (gyp.common.uniquer(deps), gyp.common.uniquer(link_deps))