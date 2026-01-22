class MemoryQuotaExceededException(YaqlException):

    def __init__(self):
        super(MemoryQuotaExceededException, self).__init__('Expression consumed too much memory')