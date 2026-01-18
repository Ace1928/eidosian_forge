import copy
def writebyproteinrec(outprotrec, handle, fields=GAF20FIELDS):
    """Write a list of GAF records to an output stream.

    Caller should know the  format version. Default: gaf-2.0
    If header has a value, then it is assumed this is the first record,
    a header is written. Typically the list is the one read by fafbyproteinrec, which
    contains all consecutive lines with the same DB_Object_ID
    """
    for outrec in outprotrec:
        writerec(outrec, handle, fields=fields)