import re
from Bio import File
Parse missing residue remarks.

    Returns a dictionary describing the missing residue.
    The specification for REMARK 465 at
    http://www.wwpdb.org/documentation/file-format-content/format33/remarks2.html#REMARK%20465
    only gives templates, but does not say they have to be followed.
    So we assume that not all pdb-files with a REMARK 465 can be understood.

    Returns a dictionary with the following keys:
    "model", "res_name", "chain", "ssseq", "insertion"
    