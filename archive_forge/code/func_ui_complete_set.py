import inspect
import re
import six
def ui_complete_set(self, parameters, text, current_param):
    """
        Parameter auto-completion method for user command set.
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
            group_params = self.list_group_params(group, writable=True)
            if current_param in group_params:
                p_def = self.get_group_param(group, current_param)
                type_method = self.get_type_method(p_def['type'])
                type_enum = type_method(enum=True)
                if type_enum is not None:
                    type_enum = [item for item in type_enum if item.startswith(text)]
                    completions.extend(type_enum)
            else:
                group_params = [param + '=' for param in group_params if param.startswith(text) if param not in parameters]
                if group_params:
                    completions.extend(group_params)
    if len(completions) == 1 and (not completions[0].endswith('=')):
        completions = [completions[0] + ' ']
    self.shell.log.debug('Returning completions %s.' % str(completions))
    return completions