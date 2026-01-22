class rd:
    """RD (reads), store a read with its name, sequence etc.

    The location and strand each read is mapped to is held in the AF lines.
    """

    def __init__(self):
        """Initialize the class."""
        self.name = ''
        self.padded_bases = None
        self.info_items = None
        self.read_tags = None
        self.sequence = ''