from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class SecretReference(dict):
    """
        Secret reference to be used as part of a :py:class:`ContainerSpec`.
        Describes how a secret is made accessible inside the service's
        containers.

        Args:
            secret_id (string): Secret's ID
            secret_name (string): Secret's name as defined at its creation.
            filename (string): Name of the file containing the secret. Defaults
                to the secret's name if not specified.
            uid (string): UID of the secret file's owner. Default: 0
            gid (string): GID of the secret file's group. Default: 0
            mode (int): File access mode inside the container. Default: 0o444
    """

    @check_resource('secret_id')
    def __init__(self, secret_id, secret_name, filename=None, uid=None, gid=None, mode=292):
        self['SecretName'] = secret_name
        self['SecretID'] = secret_id
        self['File'] = {'Name': filename or secret_name, 'UID': uid or '0', 'GID': gid or '0', 'Mode': mode}