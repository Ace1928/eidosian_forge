from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetRemoteCommand(container, command):
    """Assemble the remote command list given user-supplied args.

  If a container argument is supplied, run
  `sudo docker exec -i[t] CONTAINER_ID COMMAND [ARGS...]` on the remote.

  Args:
    container: str or None, name of container to enter during connection.
    command: [str] or None, the remote command to execute. If no command is
      given, allocate a TTY.

  Returns:
    [str] or None, Remote command to run or None if no command.
  """
    if container:
        args = command or ['/bin/sh']
        flags = '-i' if command else '-it'
        return ['sudo', 'docker', 'exec', flags, container] + args
    if command:
        return command
    return None