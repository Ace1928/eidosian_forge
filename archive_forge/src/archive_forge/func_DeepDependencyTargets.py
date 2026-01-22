import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def DeepDependencyTargets(target_dicts, roots):
    """Returns the recursive list of target dependencies."""
    dependencies = set()
    pending = set(roots)
    while pending:
        r = pending.pop()
        if r in dependencies:
            continue
        dependencies.add(r)
        spec = target_dicts[r]
        pending.update(set(spec.get('dependencies', [])))
        pending.update(set(spec.get('dependencies_original', [])))
    return list(dependencies - set(roots))