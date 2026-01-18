import copy
def writerec(outrec, handle, fields=GAF20FIELDS):
    """Write a single UniProt-GOA record to an output stream.

    Caller should know the  format version. Default: gaf-2.0
    If header has a value, then it is assumed this is the first record,
    a header is written.
    """
    outstr = ''
    for field in fields[:-1]:
        if isinstance(outrec[field], list):
            for subfield in outrec[field]:
                outstr += subfield + '|'
            outstr = outstr[:-1] + '\t'
        else:
            outstr += outrec[field] + '\t'
    outstr += outrec[fields[-1]] + '\n'
    handle.write(outstr)