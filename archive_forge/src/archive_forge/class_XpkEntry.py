class XpkEntry:
    """Provide dictionary access to single entry from nmrview .xpk file.

    This class is suited for handling single lines of non-header data
    from an nmrview .xpk file. This class provides methods for extracting
    data by the field name which is listed in the last line of the
    peaklist header.

    Parameters
    ----------
    xpkentry : str
        The line from an nmrview .xpk file.
    xpkheadline : str
        The line from the header file that gives the names of the entries.
        This is typically the sixth line of the header, 1-origin.

    Attributes
    ----------
    fields : dict
        Dictionary of fields where key is in header line, value is an entry.
        Variables are accessed by either their name in the header line as in
        self.field["H1.P"] will return the H1.P entry for example.
        self.field["entrynum"] returns the line number (1st field of line)

    """

    def __init__(self, entry, headline):
        """Initialize the class."""
        datlist = entry.split()
        headlist = headline.split()
        self.fields = dict(zip(headlist, datlist[1:]))
        try:
            self.fields['entrynum'] = datlist[0]
        except IndexError:
            pass