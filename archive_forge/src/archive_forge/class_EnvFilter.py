import os
import re
import shutil
import sys
class EnvFilter(CommandFilter):
    """Specific filter for the env utility.

    Behaves like CommandFilter, except that it handles
    leading env A=B.. strings appropriately.
    """

    def _extract_env(self, arglist):
        """Extract all leading NAME=VALUE arguments from arglist."""
        envs = set()
        for arg in arglist:
            if '=' not in arg:
                break
            envs.add(arg.partition('=')[0])
        return envs

    def __init__(self, exec_path, run_as, *args):
        super(EnvFilter, self).__init__(exec_path, run_as, *args)
        env_list = self._extract_env(self.args)
        if 'env' in exec_path and len(env_list) < len(self.args):
            self.exec_path = self.args[len(env_list)]

    def match(self, userargs):
        if userargs[0] == 'env':
            userargs.pop(0)
        if len(userargs) < len(self.args):
            return False
        user_envs = self._extract_env(userargs)
        filter_envs = self._extract_env(self.args)
        user_command = userargs[len(user_envs):len(user_envs) + 1]
        return super(EnvFilter, self).match(user_command) and len(filter_envs) and (user_envs == filter_envs)

    def exec_args(self, userargs):
        args = userargs[:]
        if args[0] == 'env':
            args.pop(0)
        while args and '=' in args[0]:
            args.pop(0)
        return args

    def get_command(self, userargs, exec_dirs=[]):
        to_exec = self.get_exec(exec_dirs=exec_dirs) or self.exec_path
        return [to_exec] + self.exec_args(userargs)[1:]

    def get_environment(self, userargs):
        env = os.environ.copy()
        if userargs[0] == 'env':
            userargs.pop(0)
        for a in userargs:
            env_name, equals, env_value = a.partition('=')
            if not equals:
                break
            if env_name and env_value:
                env[env_name] = env_value
        return env