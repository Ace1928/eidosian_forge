from libcloud.common.base import BaseDriver, ConnectionUserAndKey
from libcloud.backup.types import BackupTargetType
class BackupTarget:
    """
    A backup target
    """

    def __init__(self, id, name, address, type, driver, extra=None):
        """
        :param id: Target id
        :type id: ``str``

        :param name: Name of the target
        :type name: ``str``

        :param address: Hostname, FQDN, IP, file path etc.
        :type address: ``str``

        :param type: Backup target type (Physical, Virtual, ...).
        :type type: :class:`.BackupTargetType`

        :param driver: BackupDriver instance.
        :type driver: :class:`.BackupDriver`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``
        """
        self.id = str(id) if id else None
        self.name = name
        self.address = address
        self.type = type
        self.driver = driver
        self.extra = extra or {}

    def update(self, name=None, address=None, extra=None):
        return self.driver.update_target(target=self, name=name, address=address, extra=extra)

    def delete(self):
        return self.driver.delete_target(target=self)

    def _get_numeric_id(self):
        target_id = self.id
        if target_id.isdigit():
            target_id = int(target_id)
        return target_id

    def __repr__(self):
        return '<Target: id=%s, name=%s, address=%stype=%s, provider=%s ...>' % (self.id, self.name, self.address, self.type, self.driver.name)