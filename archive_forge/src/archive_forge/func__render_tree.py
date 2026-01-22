import inspect
import re
import six
def _render_tree(self, root, margin=None, depth=None, do_list=False):
    """
        Renders an ascii representation of a tree of ConfigNodes.
        @param root: The root node of the tree
        @type root: ConfigNode
        @param margin: Format of the left margin to use for children.
        True results in a pipe, and False results in no pipe.
        Used for recursion only.
        @type margin: list
        @param depth: The maximum depth of nodes to display, None means
        infinite.
        @type depth: None or int
        @param do_list: Return two lists, one with each line text
        representation, the other with the corresponding paths.
        @type do_list: bool
        @return: An ascii tree representation or (lines, paths).
        @rtype: str
        """
    lines = []
    paths = []
    node_length = 2
    node_shift = 2
    level = root.path.rstrip('/').count('/')
    if margin is None:
        margin = [0]
        root_call = True
    else:
        root_call = False
    if do_list:
        color = None
    elif not level % 3:
        color = None
    elif not (level - 1) % 3:
        color = 'blue'
    else:
        color = 'magenta'
    if do_list:
        styles = None
    elif root_call:
        styles = ['bold', 'underline']
    else:
        styles = ['bold']
    if do_list:
        name = root.name
    else:
        name = self.shell.con.render_text(root.name, color, styles=styles)
    name_len = len(root.name)
    description, is_healthy = root.summary()
    if not description:
        if is_healthy is True:
            description = 'OK'
        elif is_healthy is False:
            description = 'ERROR'
        else:
            description = '...'
    description_len = len(description) + 3
    if do_list:
        summary = '['
    else:
        summary = self.shell.con.render_text(' [', styles=['bold'])
    if is_healthy is True:
        if do_list:
            summary += description
        else:
            summary += self.shell.con.render_text(description, 'green')
    elif is_healthy is False:
        if do_list:
            summary += description
        else:
            summary += self.shell.con.render_text(description, 'red', styles=['bold'])
    else:
        summary += description
    if do_list:
        summary += ']'
    else:
        summary += self.shell.con.render_text(']', styles=['bold'])

    def sorting_keys(s):
        m = re.search('(.*?)(\\d+$)', str(s))
        if m:
            return (m.group(1), int(m.group(2)))
        else:
            return (str(s), 0)
    children = sorted(root.children, key=sorting_keys)
    line = ''
    for pipe in margin[:-1]:
        if pipe:
            line = line + '|'.ljust(node_shift)
        else:
            line = line + ''.ljust(node_shift)
    if self.shell.prefs['tree_round_nodes']:
        node_char = 'o'
    else:
        node_char = '+'
    line += node_char.ljust(node_length, '-')
    line += ' '
    margin_len = len(line)
    pad = (self.shell.con.get_width() - 1 - description_len - margin_len - name_len) * '.'
    if not do_list:
        pad = self.shell.con.render_text(pad, color)
    line += name
    if self.shell.prefs['tree_status_mode']:
        line += ' %s%s' % (pad, summary)
    lines.append(line)
    paths.append(root.path)
    if root_call and (not self.shell.prefs['tree_show_root']) and (not do_list):
        tree = ''
        for child in children:
            tree = tree + self._render_tree(child, [False], depth)
    else:
        tree = line + '\n'
        if depth is None or depth > 0:
            if depth is not None:
                depth = depth - 1
            for i in range(len(children)):
                margin.append(i < len(children) - 1)
                if do_list:
                    new_lines, new_paths = self._render_tree(children[i], margin, depth, do_list=True)
                    lines.extend(new_lines)
                    paths.extend(new_paths)
                else:
                    tree = tree + self._render_tree(children[i], margin, depth)
                margin.pop()
    if root_call:
        if do_list:
            return (lines, paths)
        else:
            return tree[:-1]
    elif do_list:
        return (lines, paths)
    else:
        return tree