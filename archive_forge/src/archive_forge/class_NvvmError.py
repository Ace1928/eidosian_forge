class NvvmError(Exception):

    def __str__(self):
        return '\n'.join(map(str, self.args))