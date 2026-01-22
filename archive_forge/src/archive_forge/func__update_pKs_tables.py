def _update_pKs_tables(self):
    """Update pKs tables with seq specific values for N- and C-termini."""
    pos_pKs = positive_pKs.copy()
    neg_pKs = negative_pKs.copy()
    nterm, cterm = (self.sequence[0], self.sequence[-1])
    if nterm in pKnterminal:
        pos_pKs['Nterm'] = pKnterminal[nterm]
    if cterm in pKcterminal:
        neg_pKs['Cterm'] = pKcterminal[cterm]
    return (pos_pKs, neg_pKs)