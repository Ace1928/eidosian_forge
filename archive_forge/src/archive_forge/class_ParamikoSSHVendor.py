import paramiko
import paramiko.client
class ParamikoSSHVendor:

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run_command(self, host, command, username=None, port=None, password=None, pkey=None, key_filename=None, **kwargs):
        client = paramiko.SSHClient()
        connection_kwargs = {'hostname': host}
        connection_kwargs.update(self.kwargs)
        if username:
            connection_kwargs['username'] = username
        if port:
            connection_kwargs['port'] = port
        if password:
            connection_kwargs['password'] = password
        if pkey:
            connection_kwargs['pkey'] = pkey
        if key_filename:
            connection_kwargs['key_filename'] = key_filename
        connection_kwargs.update(kwargs)
        policy = paramiko.client.MissingHostKeyPolicy()
        client.set_missing_host_key_policy(policy)
        client.connect(**connection_kwargs)
        channel = client.get_transport().open_session()
        channel.exec_command(command)
        return _ParamikoWrapper(client, channel)