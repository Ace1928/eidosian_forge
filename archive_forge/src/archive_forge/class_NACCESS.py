import os
import tempfile
import shutil
import subprocess
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.AbstractPropertyMap import (
class NACCESS(AbstractResiduePropertyMap):
    """Define NACCESS class for residue properties map."""

    def __init__(self, model, pdb_file=None, naccess_binary='naccess', tmp_directory='/tmp'):
        """Initialize the class."""
        res_data, atm_data = run_naccess(model, pdb_file, naccess=naccess_binary, temp_path=tmp_directory)
        naccess_dict = process_rsa_data(res_data)
        property_dict = {}
        property_keys = []
        property_list = []
        for chain in model:
            chain_id = chain.get_id()
            for res in chain:
                res_id = res.get_id()
                if (chain_id, res_id) in naccess_dict:
                    item = naccess_dict[chain_id, res_id]
                    res_name = item['res_name']
                    assert res_name == res.get_resname()
                    property_dict[chain_id, res_id] = item
                    property_keys.append((chain_id, res_id))
                    property_list.append((res, item))
                    res.xtra['EXP_NACCESS'] = item
                else:
                    pass
        AbstractResiduePropertyMap.__init__(self, property_dict, property_keys, property_list)