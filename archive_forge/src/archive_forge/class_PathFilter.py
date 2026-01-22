import os
import re
import shutil
import sys
class PathFilter(CommandFilter):
    """Command filter checking that path arguments are within given dirs

        One can specify the following constraints for command arguments:
            1) pass     - pass an argument as is to the resulting command
            2) some_str - check if an argument is equal to the given string
            3) abs path - check if a path argument is within the given base dir

        A typical rootwrapper filter entry looks like this:
            # cmdname: filter name, raw command, user, arg_i_constraint [, ...]
            chown: PathFilter, /bin/chown, root, nova, /var/lib/images

    """

    def match(self, userargs):
        if not userargs or len(userargs) < 2:
            return False
        arguments = userargs[1:]
        equal_args_num = len(self.args) == len(arguments)
        exec_is_valid = super(PathFilter, self).match(userargs)
        args_equal_or_pass = all((arg == 'pass' or arg == value for arg, value in zip(self.args, arguments) if not os.path.isabs(arg)))
        paths_are_within_base_dirs = all((os.path.commonprefix([arg, realpath(value)]) == arg for arg, value in zip(self.args, arguments) if os.path.isabs(arg)))
        return equal_args_num and exec_is_valid and args_equal_or_pass and paths_are_within_base_dirs

    def get_command(self, userargs, exec_dirs=None):
        exec_dirs = exec_dirs or []
        command, arguments = (userargs[0], userargs[1:])
        args = [realpath(value) if os.path.isabs(arg) else value for arg, value in zip(self.args, arguments)]
        return super(PathFilter, self).get_command([command] + args, exec_dirs)