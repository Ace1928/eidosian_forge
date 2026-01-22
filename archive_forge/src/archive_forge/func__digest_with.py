from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
def _digest_with(self, enzyme):
    cuts = []
    all_seq_cuts = []
    for seq in self.sequences:
        seq_cuts = [cut - enzyme.fst5 for cut in enzyme.search(seq)]
        all_seq_cuts.extend(seq_cuts)
        cuts.append(seq_cuts)
    all_seq_cuts = sorted(set(all_seq_cuts))
    for cut in all_seq_cuts:
        cuts_in = []
        blocked_in = []
        for i, seq in enumerate(self.sequences):
            if cut in cuts[i]:
                cuts_in.append(i)
            else:
                blocked_in.append(i)
        if cuts_in and blocked_in:
            self.dcuts.append(DifferentialCutsite(start=cut, enzyme=enzyme, cuts_in=cuts_in, blocked_in=blocked_in))