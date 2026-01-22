from Bio.Application import _Option, AbstractCommandline, _Switch
def _input_type_checker(self, command):
    return command in ('asn1_bin', 'asn1_txt', 'blastdb', 'fasta')