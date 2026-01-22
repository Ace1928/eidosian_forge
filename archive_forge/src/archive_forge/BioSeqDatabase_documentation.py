import os
from . import BioSeq
from . import Loader
from . import DBUtils
Load a set of SeqRecords into the BioSQL database.

        record_iterator is either a list of SeqRecord objects, or an
        Iterator object that returns SeqRecord objects (such as the
        output from the Bio.SeqIO.parse() function), which will be
        used to populate the database.

        fetch_NCBI_taxonomy is boolean flag allowing or preventing
        connection to the taxonomic database on the NCBI server
        (via Bio.Entrez) to fetch a detailed taxonomy for each
        SeqRecord.

        Example::

            from Bio import SeqIO
            count = db.load(SeqIO.parse(open(filename), format))

        Returns the number of records loaded.
        