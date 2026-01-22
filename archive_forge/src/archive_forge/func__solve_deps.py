from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def _solve_deps(collector_names, all_fact_subsets):
    unresolved = collector_names.copy()
    solutions = collector_names.copy()
    while True:
        unresolved = find_unresolved_requires(solutions, all_fact_subsets)
        if unresolved == set():
            break
        new_names = resolve_requires(unresolved, all_fact_subsets)
        solutions.update(new_names)
    return solutions