from libcloud.backup.base import BackupDriver
class DummyBackupDriver(BackupDriver):
    """
    Dummy Backup driver.

    >>> from libcloud.backup.drivers.dummy import DummyBackupDriver
    >>> driver = DummyBackupDriver('key', 'secret')
    >>> driver.name
    'Dummy Backup Provider'
    """
    name = 'Dummy Backup Provider'
    website = 'http://example.com'

    def __init__(self, api_key, api_secret):
        """
        :param    api_key:    API key or username to used (required)
        :type     api_key:    ``str``

        :param    api_secret: Secret password to be used (required)
        :type     api_secret: ``str``

        :rtype: ``None``
        """