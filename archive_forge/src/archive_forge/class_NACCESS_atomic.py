import os
import tempfile
import shutil
import subprocess
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.AbstractPropertyMap import (
class NACCESS_atomic(AbstractAtomPropertyMap):
    """Define NACCESS atomic class for atom properties map."""

    def __init__(self, model, pdb_file=None, naccess_binary='naccess', tmp_directory='/tmp'):
        """Initialize the class."""
        res_data, atm_data = run_naccess(model, pdb_file, naccess=naccess_binary, temp_path=tmp_directory)
        self.naccess_atom_dict = process_asa_data(atm_data)
        property_dict = {}
        property_keys = []
        property_list = []
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                res_id = residue.get_id()
                for atom in residue:
                    atom_id = atom.get_id()
                    full_id = (chain_id, res_id, atom_id)
                    if full_id in self.naccess_atom_dict:
                        asa = self.naccess_atom_dict[full_id]
                        property_dict[full_id] = asa
                        property_keys.append(full_id)
                        property_list.append((atom, asa))
                        atom.xtra['EXP_NACCESS'] = asa
        AbstractAtomPropertyMap.__init__(self, property_dict, property_keys, property_list)