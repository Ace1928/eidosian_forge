import inspect
import re
import six
def ui_command_get(self, group=None, *parameter):
    """
        Gets the value of one or more configuration parameters in the given
        group.

        Run with no parameter nor group to list all available groups, or
        with just a group name to list all available parameters within that
        group.

        Example: get global color_mode loglevel_console

        SEE ALSO
        ========
        set
        """
    if group is None:
        self.shell.con.epy_write('\n                                     AVAILABLE CONFIGURATION GROUPS\n                                     ==============================\n                                     %s\n                                     ' % ' '.join(self.list_config_groups()))
    elif not parameter:
        if group not in self.list_config_groups():
            raise ExecutionError('Unknown configuration group: %s' % group)
        section = '%s CONFIG GROUP' % group.upper()
        underline1 = ''.ljust(len(section), '=')
        parameters = ''
        params = [self.get_group_param(group, p_name) for p_name in self.list_group_params(group)]
        for p_def in params:
            group_getter = self.get_group_getter(group)
            value = group_getter(p_def['name'])
            type_method = self.get_type_method(p_def['type'])
            value = type_method(value, reverse=True)
            param = '%s=%s' % (p_def['name'], value)
            if p_def['writable'] is False:
                param += ' [ro]'
            underline2 = ''.ljust(len(param), '-')
            parameters += '%s\n%s\n%s\n\n' % (param, underline2, p_def['description'])
        self.shell.con.epy_write('%s\n%s\n%s\n' % (section, underline1, parameters))
    elif group not in self.list_config_groups():
        raise ExecutionError('Unknown configuration group: %s' % group)
    for param in parameter:
        if param not in self.list_group_params(group):
            raise ExecutionError("No parameter '%s' in group '%s'." % (param, group))
        self.shell.log.debug("About to get the parameter's value.")
        group_getter = self.get_group_getter(group)
        value = group_getter(param)
        p_def = self.get_group_param(group, param)
        type_method = self.get_type_method(p_def['type'])
        value = type_method(value, reverse=True)
        if p_def['writable']:
            writable = ''
        else:
            writable = '[ro]'
        self.shell.con.display('%s=%s %s' % (param, value, writable))