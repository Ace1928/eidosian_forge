import os
import re
import shelve
import sys
import nltk.data
def val_load(db):
    """
    Load a ``Valuation`` from a persistent database.

    :param db: name of file from which data is read.
               The suffix '.db' should be omitted from the name.
    :type db: str
    """
    dbname = db + '.db'
    if not os.access(dbname, os.R_OK):
        sys.exit('Cannot read file: %s' % dbname)
    else:
        db_in = shelve.open(db)
        from nltk.sem import Valuation
        val = Valuation(db_in)
        return val