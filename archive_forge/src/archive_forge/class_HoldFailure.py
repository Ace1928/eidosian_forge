import errno
class HoldFailure(MultipleOperationsFailure):
    message = 'Placement of hold(s) failed for one or more reasons'

    def __init__(self, errors, suppressed_count):
        super(HoldFailure, self).__init__(errors, suppressed_count)