import getpass
import logging
import urllib.parse
import smart_open.utils
def _connect_ssh(hostname, username, port, password, transport_params):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = transport_params.get('connect_kwargs', {}).copy()
    if 'key_filename' not in kwargs:
        kwargs.setdefault('password', password)
    kwargs.setdefault('username', username)
    ssh.connect(hostname, port, **kwargs)
    return ssh