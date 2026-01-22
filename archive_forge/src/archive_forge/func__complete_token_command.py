import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def _complete_token_command(self, text, path, command):
    """
        Completes a partial command token, which could also be the beginning
        of a path.
        @param path: Path of the target ConfigNode.
        @type path: str
        @param command: The command (if any) found by the parser.
        @type command: str
        @param text: Current text being typed by the user.
        @type text: str
        @return: Possible completions for the token.
        @rtype: list of str
        """
    completions = []
    target = self._current_node.get_node(path)
    commands = target.list_commands()
    self.log.debug('Completing command token among %s' % str(commands))
    for command in commands:
        if command.startswith(text):
            completions.append(command)
    if len(completions) == 1:
        completions[0] = completions[0] + ' '
    if not path:
        path_completions = [child.name + '/' for child in self._current_node.children if child.name.startswith(text)]
        if not text:
            path_completions.append('/')
            if len(self._current_node.children) > 1:
                path_completions.append('* ')
        if path_completions:
            if completions:
                self._current_token = self.con.render_text('path', self.prefs['color_path']) + '|' + self.con.render_text('command', self.prefs['color_command'])
            else:
                self._current_token = self.con.render_text('path', self.prefs['color_path'])
        else:
            self._current_token = self.con.render_text('command', self.prefs['color_command'])
        if len(path_completions) == 1 and (not path_completions[0][-1] in [' ', '*']) and (not self._current_node.get_node(path_completions[0]).children):
            path_completions[0] = path_completions[0] + ' '
        completions.extend(path_completions)
    else:
        self._current_token = self.con.render_text('command', self.prefs['color_command'])
    bookmarks = ['@' + bookmark for bookmark in self.prefs['bookmarks'] if bookmark.startswith('%s' % text.lstrip('@'))]
    self.log.debug('Found bookmarks %s.' % str(bookmarks))
    if bookmarks:
        completions.extend(bookmarks)
    return completions