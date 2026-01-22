import errno
class SnapshotDestructionFailure(MultipleOperationsFailure):
    message = 'Destruction of snapshot(s) failed for one or more reasons'

    def __init__(self, errors, suppressed_count):
        super(SnapshotDestructionFailure, self).__init__(errors, suppressed_count)