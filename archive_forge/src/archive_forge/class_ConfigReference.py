from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class ConfigReference(dict):
    """
        Config reference to be used as part of a :py:class:`ContainerSpec`.
        Describes how a config is made accessible inside the service's
        containers.

        Args:
            config_id (string): Config's ID
            config_name (string): Config's name as defined at its creation.
            filename (string): Name of the file containing the config. Defaults
                to the config's name if not specified.
            uid (string): UID of the config file's owner. Default: 0
            gid (string): GID of the config file's group. Default: 0
            mode (int): File access mode inside the container. Default: 0o444
    """

    @check_resource('config_id')
    def __init__(self, config_id, config_name, filename=None, uid=None, gid=None, mode=292):
        self['ConfigName'] = config_name
        self['ConfigID'] = config_id
        self['File'] = {'Name': filename or config_name, 'UID': uid or '0', 'GID': gid or '0', 'Mode': mode}