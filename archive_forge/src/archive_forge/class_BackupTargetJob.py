from libcloud.common.base import BaseDriver, ConnectionUserAndKey
from libcloud.backup.types import BackupTargetType
class BackupTargetJob:
    """
    A backup target job
    """

    def __init__(self, id, status, progress, target, driver, extra=None):
        """
        :param id: Job id
        :type id: ``str``

        :param status: Status of the job
        :type status: :class:`BackupTargetJobStatusType`

        :param progress: Progress of the job, as a percentage
        :type progress: ``int``

        :param target: BackupTarget instance.
        :type target: :class:`.BackupTarget`

        :param driver: BackupDriver instance.
        :type driver: :class:`.BackupDriver`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``
        """
        self.id = str(id) if id else None
        self.status = status
        self.progress = progress
        self.target = target
        self.driver = driver
        self.extra = extra or {}

    def cancel(self):
        return self.driver.cancel_target_job(job=self)

    def suspend(self):
        return self.driver.suspend_target_job(job=self)

    def resume(self):
        return self.driver.resume_target_job(job=self)

    def __repr__(self):
        return '<Job: id=%s, status=%s, progress=%starget=%s, provider=%s ...>' % (self.id, self.status, self.progress, self.target.id, self.driver.name)