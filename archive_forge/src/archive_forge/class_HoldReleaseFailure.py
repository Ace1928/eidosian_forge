import errno
class HoldReleaseFailure(MultipleOperationsFailure):
    message = 'Release of hold(s) failed for one or more reasons'

    def __init__(self, errors, suppressed_count):
        super(HoldReleaseFailure, self).__init__(errors, suppressed_count)