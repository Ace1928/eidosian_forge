import errno
class BookmarkDestructionFailure(MultipleOperationsFailure):
    message = 'Destruction of bookmark(s) failed for one or more reasons'

    def __init__(self, errors, suppressed_count):
        super(BookmarkDestructionFailure, self).__init__(errors, suppressed_count)