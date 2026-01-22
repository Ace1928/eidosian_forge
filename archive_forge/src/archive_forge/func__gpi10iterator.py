import copy
def _gpi10iterator(handle):
    """Read GPI 1.0 format files (PRIVATE).

    This iterator is used to read a gp_information.goa_uniprot
    file which is in the GPI 1.0 format.
    """
    for inline in handle:
        if inline[0] == '!':
            continue
        inrec = inline.rstrip('\n').split('\t')
        if len(inrec) == 1:
            continue
        inrec[5] = inrec[5].split('|')
        inrec[8] = inrec[8].split('|')
        yield dict(zip(GPI10FIELDS, inrec))