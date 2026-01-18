import os
import re
import shelve
import sys
import nltk.data
def val_dump(rels, db):
    """
    Make a ``Valuation`` from a list of relation metadata bundles and dump to
    persistent database.

    :param rels: bundle of metadata needed for constructing a concept
    :type rels: list of dict
    :param db: name of file to which data is written.
               The suffix '.db' will be automatically appended.
    :type db: str
    """
    concepts = process_bundle(rels).values()
    valuation = make_valuation(concepts, read=True)
    db_out = shelve.open(db, 'n')
    db_out.update(valuation)
    db_out.close()