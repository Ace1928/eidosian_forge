from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
Parse comments (PRIVATE).

            Comment fields are very heterogeneus. each type has his own (frequently mutated) schema.
            To store all the contained data, more complex data structures are needed, such as
            annotated dictionaries. This is left to end user, by optionally setting:

            return_raw_comments=True

            The original XML is returned in the annotation fields.

            Available comment types at december 2009:
             - "allergen"
             - "alternative products"
             - "biotechnology"
             - "biophysicochemical properties"
             - "catalytic activity"
             - "caution"
             - "cofactor"
             - "developmental stage"
             - "disease"
             - "domain"
             - "disruption phenotype"
             - "enzyme regulation"
             - "function"
             - "induction"
             - "miscellaneous"
             - "pathway"
             - "pharmaceutical"
             - "polymorphism"
             - "PTM"
             - "RNA editing"
             - "similarity"
             - "subcellular location"
             - "sequence caution"
             - "subunit"
             - "tissue specificity"
             - "toxic dose"
             - "online information"
             - "mass spectrometry"
             - "interaction"

            