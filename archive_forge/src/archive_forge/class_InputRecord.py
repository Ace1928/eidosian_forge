class InputRecord:
    """Represent the input file into the primersearch program.

    This makes it easy to add primer information and write it out to the
    simple primer file format.
    """

    def __init__(self):
        """Initialize the class."""
        self.primer_info = []

    def __str__(self):
        """Summarize the primersearch input record as a string."""
        output = ''
        for name, primer1, primer2 in self.primer_info:
            output += f'{name} {primer1} {primer2}\n'
        return output

    def add_primer_set(self, primer_name, first_primer_seq, second_primer_seq):
        """Add primer information to the record."""
        self.primer_info.append((primer_name, first_primer_seq, second_primer_seq))