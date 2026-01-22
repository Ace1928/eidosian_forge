from tempest.lib import exceptions
class SnapshotBuildErrorException(exceptions.TempestException):
    message = 'Snapshot %(snapshot)s failed to build and is in ERROR status.'