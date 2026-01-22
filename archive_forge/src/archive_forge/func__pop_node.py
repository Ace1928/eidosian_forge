from . import errors
from . import graph as _mod_graph
from . import revision as _mod_revision
def _pop_node(self):
    """Pop the top node off the stack

        The node is appended to the sorted output.
        """
    node_name = self._node_name_stack.pop()
    merge_depth = self._node_merge_depth_stack.pop()
    first_child = self._first_child_stack.pop()
    self._left_subtree_pushed_stack.pop()
    self._pending_parents_stack.pop()
    parents = self._original_graph[node_name]
    parent_revno = None
    if parents:
        try:
            parent_revno = self._revnos[parents[0]][0]
        except KeyError:
            pass
    if parent_revno is not None:
        if not first_child:
            base_revno = parent_revno[0]
            branch_count = self._revno_to_branch_count.get(base_revno, 0)
            branch_count += 1
            self._revno_to_branch_count[base_revno] = branch_count
            revno = (parent_revno[0], branch_count, 1)
        else:
            revno = parent_revno[:-1] + (parent_revno[-1] + 1,)
    else:
        root_count = self._revno_to_branch_count.get(0, 0)
        root_count = self._revno_to_branch_count.get(0, -1)
        root_count += 1
        if root_count:
            revno = (0, root_count, 1)
        else:
            revno = (1,)
        self._revno_to_branch_count[0] = root_count
    self._revnos[node_name][0] = revno
    self._completed_node_names.add(node_name)
    self._scheduled_nodes.append((node_name, merge_depth, self._revnos[node_name][0]))
    return node_name