import inspect
import re
import six
def ui_command_ls(self, path=None, depth=None):
    """
        Display either the nodes tree relative to path or to the current node.

        PARAMETERS
        ==========

        path
        ----
        The path to display the nodes tree of. Can be an absolute path, a
        relative path or a bookmark.

        depth
        -----
        The depth parameter limits the maximum depth of the tree to display.
        If set to 0, then the complete tree will be displayed (the default).

        SEE ALSO
        ========
        cd bookmarks
        """
    try:
        target = self.get_node(path)
    except ValueError as msg:
        raise ExecutionError(str(msg))
    if depth is None:
        depth = self.shell.prefs['tree_max_depth']
    try:
        depth = int(depth)
    except ValueError:
        raise ExecutionError('The tree depth must be a number.')
    if depth == 0:
        depth = None
    tree = self._render_tree(target, depth=depth)
    self.shell.con.display(tree)