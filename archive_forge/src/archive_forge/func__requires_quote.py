import re
from collections import defaultdict
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import Select, StructureIO
def _requires_quote(self, val):
    if ' ' in val or "'" in val or '"' in val or (val[0] in ['_', '#', '$', '[', ']', ';']) or val.startswith(('data_', 'save_')) or (val in ['loop_', 'stop_', 'global_']):
        return True
    else:
        return False