def add_primer_set(self, primer_name, first_primer_seq, second_primer_seq):
    """Add primer information to the record."""
    self.primer_info.append((primer_name, first_primer_seq, second_primer_seq))