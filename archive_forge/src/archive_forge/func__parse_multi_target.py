import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_multi_target(self, targets):
    """validate multi targets that defined between parentheses()"""
    if not targets:
        self.dist_fatal("empty multi-target '()'")
    if not all([self.feature_is_exist(tar) for tar in targets]):
        self.dist_fatal('invalid target name in multi-target', targets)
    if not all([tar in self.parse_baseline_names or tar in self.parse_dispatch_names for tar in targets]):
        return None
    targets = self.feature_ahead(targets)
    if not targets:
        return None
    targets = self.feature_sorted(targets)
    targets = tuple(targets)
    return targets