import gyp.common
import json
import os
import posixpath
def _AddCompileTargets(target, roots, add_if_no_ancestor, result):
    """Recurses through all targets that depend on |target|, adding all targets
  that need to be built (and are in |roots|) to |result|.
  roots: set of root targets.
  add_if_no_ancestor: If true and there are no ancestors of |target| then add
  |target| to |result|. |target| must still be in |roots|.
  result: targets that need to be built are added here."""
    if target.visited:
        return
    target.visited = True
    target.in_roots = target in roots
    for back_dep_target in target.back_deps:
        _AddCompileTargets(back_dep_target, roots, False, result)
        target.added_to_compile_targets |= back_dep_target.added_to_compile_targets
        target.in_roots |= back_dep_target.in_roots
        target.is_or_has_linked_ancestor |= back_dep_target.is_or_has_linked_ancestor
    if target.in_roots and (target.is_executable or (not target.added_to_compile_targets and (add_if_no_ancestor or target.requires_build)) or (target.is_static_library and add_if_no_ancestor and (not target.is_or_has_linked_ancestor))):
        print('\t\tadding to compile targets', target.name, 'executable', target.is_executable, 'added_to_compile_targets', target.added_to_compile_targets, 'add_if_no_ancestor', add_if_no_ancestor, 'requires_build', target.requires_build, 'is_static_library', target.is_static_library, 'is_or_has_linked_ancestor', target.is_or_has_linked_ancestor)
        result.add(target)
        target.added_to_compile_targets = True