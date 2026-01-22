class af:
    """AF lines, define the location of the read within the contig.

    Note attribute coru is short for complemented (C) or uncomplemented (U),
    since the strand information is stored in an ACE file using either the
    C or U character.
    """

    def __init__(self, line=None):
        """Initialize the class."""
        self.name = ''
        self.coru = None
        self.padded_start = None
        if line:
            header = line.split()
            self.name = header[1]
            self.coru = header[2]
            self.padded_start = int(header[3])