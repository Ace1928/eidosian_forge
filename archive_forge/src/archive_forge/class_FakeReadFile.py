from io import BytesIO
class FakeReadFile:
    """A file-like object that can be given predefined content and read
    like a file.  The maximum size and number of the reads is recorded."""

    def __init__(self, data):
        """Initialize the mock file object with the provided data."""
        self.data = BytesIO(data)
        self.max_read_size = None
        self.read_count = 0

    def read(self, size=-1):
        """Reads size characters from the input (or the rest of the string if
        size is -1)."""
        data = self.data.read(size)
        self.max_read_size = max(self.max_read_size or 0, len(data))
        self.read_count += 1
        return data

    def get_max_read_size(self):
        """Returns the maximum read size or None if no reads have occured."""
        return self.max_read_size

    def get_read_count(self):
        """Returns the number of calls to read."""
        return self.read_count

    def reset_read_count(self):
        """Clears the read count."""
        self.read_count = 0