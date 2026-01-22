class STSLine:
    """Store the information for one STS line from a Unigene file.

    Initialize with the text part of the STS line, or nothing.

    Attributes and descriptions (access as LOWER CASE)

    ACC=         GenBank/EMBL/DDBJ accession number of STS [optional field]
    UNISTS=      identifier in NCBI's UNISTS database
    """

    def __init__(self, text=None):
        """Initialize the class."""
        self.acc = ''
        self.unists = ''
        if text is not None:
            self.text = text
            self._init_from_text(text)

    def _init_from_text(self, text):
        parts = text.split(' ')
        for part in parts:
            key, val = part.split('=')
            setattr(self, key.lower(), val)

    def __repr__(self):
        """Return UniGene STSLine object as a string."""
        return self.text