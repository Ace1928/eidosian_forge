class file_generator(object):
    """Yield the given input (a file object) in chunks (default 64k).

    (Core)
    """

    def __init__(self, input, chunkSize=65536):
        """Initialize file_generator with file ``input`` for chunked access."""
        self.input = input
        self.chunkSize = chunkSize

    def __iter__(self):
        """Return iterator."""
        return self

    def __next__(self):
        """Return next chunk of file."""
        chunk = self.input.read(self.chunkSize)
        if chunk:
            return chunk
        else:
            if hasattr(self.input, 'close'):
                self.input.close()
            raise StopIteration()
    next = __next__

    def __del__(self):
        """Close input on descturct."""
        if hasattr(self.input, 'close'):
            self.input.close()