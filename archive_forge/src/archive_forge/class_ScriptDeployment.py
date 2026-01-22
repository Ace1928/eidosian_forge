import os
import re
import binascii
from typing import IO, List, Union, Optional, cast
from libcloud.utils.py3 import basestring
from libcloud.compute.ssh import BaseSSHClient
from libcloud.compute.base import Node
class ScriptDeployment(Deployment):
    """
    Runs an arbitrary shell script on the server.

    This step works by first writing the content of the shell script (script
    argument) in a *.sh file on a remote server and then running that file.

    If you are running a non-shell script, make sure to put the appropriate
    shebang to the top of the script. You are also advised to do that even if
    you are running a plan shell script.
    """

    def __init__(self, script, args=None, name=None, delete=False, timeout=None):
        """
        :type script: ``str``
        :keyword script: Contents of the script to run.

        :type args: ``list``
        :keyword args: Optional command line arguments which get passed to the
                       deployment script file.

        :type name: ``str``
        :keyword name: Name of the script to upload it as, if not specified,
                       a random name will be chosen.

        :type delete: ``bool``
        :keyword delete: Whether to delete the script on completion.

        :param timeout: Optional run timeout for this command.
        :type timeout: ``float``
        """
        script = self._get_string_value(argument_name='script', argument_value=script)
        self.script = script
        self.args = args or []
        self.stdout = None
        self.stderr = None
        self.exit_status = None
        self.delete = delete
        self.timeout = timeout
        self.name = name
        if self.name is None:
            random_string = ''
            random_string = binascii.hexlify(os.urandom(4))
            random_string = cast(bytes, random_string)
            random_string = random_string.decode('ascii')
            self.name = 'libcloud_deployment_%s.sh' % random_string

    def run(self, node, client):
        """
        Uploads the shell script and then executes it.

        See also :class:`Deployment.run`
        """
        self.name = cast(str, self.name)
        file_path = client.put(path=self.name, chmod=int('755', 8), contents=self.script)
        if self.name and (self.name[0] not in ['/', '\\'] and (not re.match('^\\w\\:.*$', file_path))):
            base_path = os.path.dirname(file_path)
            name = os.path.join(base_path, self.name)
        elif self.name and (self.name[0] == '\\' or re.match('^\\w\\:.*$', file_path)):
            name = file_path
        else:
            self.name = cast(str, self.name)
            name = self.name
        cmd = name
        if self.args:
            cmd = '{} {}'.format(name, ' '.join(self.args))
        else:
            cmd = name
        self.stdout, self.stderr, self.exit_status = client.run(cmd, timeout=self.timeout)
        if self.delete:
            client.delete(self.name)
        return node

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        script = self.script[:15] + '...'
        exit_status = self.exit_status
        if exit_status is not None:
            stdout = self.stdout[:30] + '...'
            stderr = self.stderr[:30] + '...'
        else:
            exit_status = "script didn't run yet"
            stdout = None
            stderr = None
        return '<ScriptDeployment script=%s, exit_status=%s, stdout=%s, stderr=%s>' % (script, exit_status, stdout, stderr)