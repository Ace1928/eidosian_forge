class ProtsimLine:
    """Store the information for one PROTSIM line from a Unigene file.

    Initialize with the text part of the PROTSIM line, or nothing.

    Attributes and descriptions (access as LOWER CASE)
    ORG=         Organism
    PROTGI=      Sequence GI of protein
    PROTID=      Sequence ID of protein
    PCT=         Percent alignment
    ALN=         length of aligned region (aa)
    """

    def __init__(self, text=None):
        """Initialize the class."""
        self.org = ''
        self.protgi = ''
        self.protid = ''
        self.pct = ''
        self.aln = ''
        if text is not None:
            self.text = text
            self._init_from_text(text)

    def _init_from_text(self, text):
        parts = text.split('; ')
        for part in parts:
            key, val = part.split('=')
            setattr(self, key.lower(), val)

    def __repr__(self):
        """Return UniGene ProtsimLine object as a string."""
        return self.text