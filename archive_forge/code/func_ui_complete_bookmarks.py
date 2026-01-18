import inspect
import re
import six
def ui_complete_bookmarks(self, parameters, text, current_param):
    """
        Parameter auto-completion method for user command bookmarks.
        @param parameters: Parameters on the command line.
        @type parameters: dict
        @param text: Current text of parameter being typed by the user.
        @type text: str
        @param current_param: Name of parameter to complete.
        @type current_param: str
        @return: Possible completions
        @rtype: list of str
        """
    if current_param == 'action':
        completions = [action for action in ['add', 'del', 'go', 'show'] if action.startswith(text)]
    elif current_param == 'bookmark':
        if 'action' in parameters:
            if parameters['action'] not in ['show', 'add']:
                completions = [mark for mark in self.shell.prefs['bookmarks'] if mark.startswith(text)]
    else:
        completions = []
    if len(completions) == 1:
        return [completions[0] + ' ']
    else:
        return completions