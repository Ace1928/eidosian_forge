import inspect
import re
import six
def ui_complete_help(self, parameters, text, current_param):
    """
        Parameter auto-completion method for user command help.
        @param parameters: Parameters on the command line.
        @type parameters: dict
        @param text: Current text of parameter being typed by the user.
        @type text: str
        @param current_param: Name of parameter to complete.
        @type current_param: str
        @return: Possible completions
        @rtype: list of str
        """
    if current_param == 'topic':
        topics = self.list_commands()
        completions = [topic for topic in topics if topic.startswith(text)]
    else:
        completions = []
    if len(completions) == 1:
        return [completions[0] + ' ']
    else:
        return completions