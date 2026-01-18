import inspect
import re
import six
def ui_command_cd(self, path=None):
    """
        Change current path to path.

        The path is constructed just like a unix path, with "/" as separator
        character, "." for the current node, ".." for the parent node.

        Suppose the nodes tree looks like this:
           +-/
           +-a0      (1)
           | +-b0    (*)
           |  +-c0
           +-a1      (3)
             +-b0
              +-c0
               +-d0  (2)

        Suppose the current node is the one marked (*) at the beginning of all
        the following examples:
            - `cd ..` takes you to the node marked (1)
            - `cd .` makes you stay in (*)
            - `cd /a1/b0/c0/d0` takes you to the node marked (2)
            - `cd ../../a1` takes you to the node marked (3)
            - `cd /a1` also takes you to the node marked (3)
            - `cd /` takes you to the root node "/"
            - `cd /a0/b0/./c0/../../../a1/.` takes you to the node marked (3)

        You can also navigate the path history with "<" and ">":
            - `cd <` takes you back one step in the path history
            - `cd >` takes you one step forward in the path history

        SEE ALSO
        ========
        ls cd
        """
    self.shell.log.debug("Changing current node to '%s'." % path)
    if self.shell.prefs['path_history'] is None:
        self.shell.prefs['path_history'] = [self.path]
        self.shell.prefs['path_history_index'] = 0
    if path == '<':
        if self.shell.prefs['path_history_index'] == 0:
            self.shell.log.info('Reached begining of path history.')
            return self
        exists = False
        while not exists:
            if self.shell.prefs['path_history_index'] > 0:
                self.shell.prefs['path_history_index'] = self.shell.prefs['path_history_index'] - 1
                index = self.shell.prefs['path_history_index']
                path = self.shell.prefs['path_history'][index]
                try:
                    target_node = self.get_node(path)
                except ValueError:
                    pass
                else:
                    exists = True
            else:
                path = '/'
                self.shell.prefs['path_history_index'] = 0
                self.shell.prefs['path_history'][0] = '/'
                exists = True
        self.shell.log.info('Taking you back to %s.' % path)
        return self.get_node(path)
    if path == '>':
        if self.shell.prefs['path_history_index'] == len(self.shell.prefs['path_history']) - 1:
            self.shell.log.info('Reached the end of path history.')
            return self
        exists = False
        while not exists:
            if self.shell.prefs['path_history_index'] < len(self.shell.prefs['path_history']) - 1:
                self.shell.prefs['path_history_index'] = self.shell.prefs['path_history_index'] + 1
                index = self.shell.prefs['path_history_index']
                path = self.shell.prefs['path_history'][index]
                try:
                    target_node = self.get_node(path)
                except ValueError:
                    pass
                else:
                    exists = True
            else:
                path = self.path
                self.shell.prefs['path_history_index'] = len(self.shell.prefs['path_history'])
                self.shell.prefs['path_history'].append(path)
                exists = True
        self.shell.log.info('Taking you back to %s.' % path)
        return self.get_node(path)
    if path is None:
        lines, paths = self._render_tree(self.get_root(), do_list=True)
        start_pos = paths.index(self.path)
        selected = self._lines_walker(lines, start_pos=start_pos)
        path = paths[selected]
    try:
        target_node = self.get_node(path)
    except ValueError as msg:
        raise ExecutionError(str(msg))
    index = self.shell.prefs['path_history_index']
    if target_node.path != self.shell.prefs['path_history'][index]:
        self.shell.prefs['path_history'] = self.shell.prefs['path_history'][:index + 1]
        self.shell.prefs['path_history'].append(target_node.path)
        self.shell.prefs['path_history_index'] = index + 1
    self.shell.log.debug('After cd, path history is: %s, index is %d' % (str(self.shell.prefs['path_history']), self.shell.prefs['path_history_index']))
    return target_node