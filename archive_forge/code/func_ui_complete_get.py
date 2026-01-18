import inspect
import re
import six
def ui_complete_get(self, parameters, text, current_param):
    """
        Parameter auto-completion method for user command get.
        @param parameters: Parameters on the command line.
        @type parameters: dict
        @param text: Current text of parameter being typed by the user.
        @type text: str
        @param current_param: Name of parameter to complete.
        @type current_param: str
        @return: Possible completions
        @rtype: list of str
        """
    completions = []
    self.shell.log.debug("Called with params=%s, text='%s', current='%s'" % (str(parameters), text, current_param))
    if current_param == 'group':
        completions = [group for group in self.list_config_groups() if group.startswith(text)]
    elif 'group' in parameters:
        group = parameters['group']
        if group in self.list_config_groups():
            group_params = [param for param in self.list_group_params(group) if param.startswith(text) if param not in parameters]
            if group_params:
                completions.extend(group_params)
    if len(completions) == 1 and (not completions[0].endswith('=')):
        completions = [completions[0] + ' ']
    self.shell.log.debug('Returning completions %s.' % str(completions))
    return completions