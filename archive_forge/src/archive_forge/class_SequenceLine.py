class SequenceLine:
    """Store the information for one SEQUENCE line from a Unigene file.

    Initialize with the text part of the SEQUENCE line, or nothing.

    Attributes and descriptions (access as LOWER CASE):
     - ACC=         GenBank/EMBL/DDBJ accession number of sequence
     - NID=         Unique nucleotide sequence identifier (gi)
     - PID=         Unique protein sequence identifier (used for non-ESTs)
     - CLONE=       Clone identifier (used for ESTs only)
     - END=         End (5'/3') of clone insert read (used for ESTs only)
     - LID=         Library ID; see Hs.lib.info for library name and tissue
     - MGC=         5' CDS-completeness indicator; if present,
       the clone associated with this sequence
       is believed CDS-complete. A value greater than 511
       is the gi of the CDS-complete mRNA matched by the EST,
       otherwise the value is an indicator of the reliability
       of the test indicating CDS completeness;
       higher values indicate more reliable CDS-completeness
       predictions.
     - SEQTYPE=     Description of the nucleotide sequence. Possible values
       are mRNA, EST and HTC.
     - TRACE=       The Trace ID of the EST sequence, as provided by NCBI
       Trace Archive

    """

    def __init__(self, text=None):
        """Initialize the class."""
        self.acc = ''
        self.nid = ''
        self.lid = ''
        self.pid = ''
        self.clone = ''
        self.image = ''
        self.is_image = False
        self.end = ''
        self.mgc = ''
        self.seqtype = ''
        self.trace = ''
        if text is not None:
            self.text = text
            self._init_from_text(text)

    def _init_from_text(self, text):
        parts = text.split('; ')
        for part in parts:
            key, val = part.split('=')
            if key == 'CLONE':
                if val[:5] == 'IMAGE':
                    self.is_image = True
                    self.image = val[6:]
            setattr(self, key.lower(), val)

    def __repr__(self):
        """Return UniGene SequenceLine object as a string."""
        return self.text