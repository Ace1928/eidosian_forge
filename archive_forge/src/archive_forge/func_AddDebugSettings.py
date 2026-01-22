import os
import re
import socket  # for gethostname
import gyp.easy_xml as easy_xml
def AddDebugSettings(self, config_name, command, environment={}, working_directory=''):
    """Adds a DebugSettings node to the user file for a particular config.

    Args:
      command: command line to run.  First element in the list is the
        executable.  All elements of the command will be quoted if
        necessary.
      working_directory: other files which may trigger the rule. (optional)
    """
    command = _QuoteWin32CommandLineArgs(command)
    abs_command = _FindCommandInPath(command[0])
    if environment and isinstance(environment, dict):
        env_list = [f'{key}="{val}"' for key, val in environment.items()]
        environment = ' '.join(env_list)
    else:
        environment = ''
    n_cmd = ['DebugSettings', {'Command': abs_command, 'WorkingDirectory': working_directory, 'CommandArguments': ' '.join(command[1:]), 'RemoteMachine': socket.gethostname(), 'Environment': environment, 'EnvironmentMerge': 'true', 'Attach': 'false', 'DebuggerType': '3', 'Remote': '1', 'RemoteCommand': '', 'HttpUrl': '', 'PDBPath': '', 'SQLDebugging': '', 'DebuggerFlavor': '0', 'MPIRunCommand': '', 'MPIRunArguments': '', 'MPIRunWorkingDirectory': '', 'ApplicationCommand': '', 'ApplicationArguments': '', 'ShimCommand': '', 'MPIAcceptMode': '', 'MPIAcceptFilter': ''}]
    if config_name not in self.configurations:
        self.AddConfig(config_name)
    self.configurations[config_name].append(n_cmd)