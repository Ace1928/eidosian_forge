import copy
def _gpa10iterator(handle):
    """Read GPA 1.0 format files (PRIVATE).

    This iterator is used to read a gp_association.*
    file which is in the GPA 1.0 format. Do not call directly. Rather,
    use the gpaiterator function.
    """
    for inline in handle:
        if inline[0] == '!':
            continue
        inrec = inline.rstrip('\n').split('\t')
        if len(inrec) == 1:
            continue
        inrec[2] = inrec[2].split('|')
        inrec[4] = inrec[4].split('|')
        inrec[6] = inrec[6].split('|')
        inrec[10] = inrec[10].split('|')
        yield dict(zip(GPA10FIELDS, inrec))