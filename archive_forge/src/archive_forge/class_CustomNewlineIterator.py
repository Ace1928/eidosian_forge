import io
class CustomNewlineIterator:
    """
    Used to iterate through files in binary mode line by line where newline != b'\\n'.

    Parameters
    ----------
    _file : file-like object
        File-like object to iterate over.
    newline : bytes
        Byte or sequence of bytes indicating line endings.
    """

    def __init__(self, _file, newline):
        self.file = _file
        self.newline = newline
        self.bytes_read = self.chunk_size = 0

    def __iter__(self):
        """
        Iterate over lines.

        Yields
        ------
        bytes
            Data from file.
        """
        buffer_size = io.DEFAULT_BUFFER_SIZE
        chunk = self.file.read(buffer_size)
        self.chunk_size = 0
        while chunk:
            self.bytes_read = 0
            self.chunk_size = len(chunk)
            lines = chunk.split(self.newline)
            for line in lines[:-1]:
                self.bytes_read += len(line) + len(self.newline)
                yield line
            chunk = self.file.read(buffer_size)
            if lines[-1]:
                chunk = lines[-1] + chunk

    def seek(self):
        """Change the stream positition to where the last returned line ends."""
        self.file.seek(self.bytes_read - self.chunk_size, 1)