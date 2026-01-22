class qa:
    """QA (read quality), including which part if any was used as the consensus."""

    def __init__(self, line=None):
        """Initialize the class."""
        self.qual_clipping_start = None
        self.qual_clipping_end = None
        self.align_clipping_start = None
        self.align_clipping_end = None
        if line:
            header = line.split()
            self.qual_clipping_start = int(header[1])
            self.qual_clipping_end = int(header[2])
            self.align_clipping_start = int(header[3])
            self.align_clipping_end = int(header[4])