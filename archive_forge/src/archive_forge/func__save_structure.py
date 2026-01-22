import itertools
from collections import defaultdict
from string import ascii_uppercase
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import Select, StructureIO
from mmtf.api.mmtf_writer import MMTFEncoder
from Bio.SeqUtils import seq1
from Bio.Data.PDBData import protein_letters_3to1_extended
def _save_structure(self, filepath, select):
    count_models, count_chains, count_groups, count_atoms = (0, 0, 0, 0)
    atom_serials = [a.serial_number for a in self.structure.get_atoms()]
    renumber_atoms = None in atom_serials
    encoder = MMTFEncoder()
    encoder.init_structure(total_num_bonds=0, total_num_atoms=0, total_num_groups=0, total_num_chains=0, total_num_models=0, structure_id=self.structure.id)
    encoder.set_xtal_info(space_group='', unit_cell=None)
    header_dict = defaultdict(str, self.structure.header)
    if header_dict['resolution'] == '':
        header_dict['resolution'] = None
    if header_dict['structure_method'] == '':
        header_dict['structure_method'] = []
    else:
        header_dict['structure_method'] = [header_dict['structure_method']]
    encoder.set_header_info(r_free=None, r_work=None, resolution=header_dict['resolution'], title=header_dict['name'], deposition_date=header_dict['deposition_date'], release_date=header_dict['release_date'], experimental_methods=header_dict['structure_method'])
    chains_per_model = []
    groups_per_chain = []
    for mi, model in enumerate(self.structure.get_models()):
        if not select.accept_model(model):
            continue
        chain_id_iterator = self._chain_id_iterator()
        count_models += 1
        encoder.set_model_info(model_id=mi, chain_count=0)
        for chain in model.get_chains():
            if not select.accept_chain(chain):
                continue
            seqs = []
            seq = ''
            prev_residue_type = ''
            prev_resname = ''
            first_chain = True
            for residue in chain.get_unpacked_list():
                if not select.accept_residue(residue):
                    continue
                count_groups += 1
                hetfield, resseq, icode = residue.get_id()
                if hetfield == ' ':
                    residue_type = 'ATOM'
                    entity_type = 'polymer'
                elif hetfield == 'W':
                    residue_type = 'HETATM'
                    entity_type = 'water'
                else:
                    residue_type = 'HETATM'
                    entity_type = 'non-polymer'
                resname = residue.get_resname()
                if residue_type != prev_residue_type or (residue_type == 'HETATM' and resname != prev_resname):
                    encoder.set_entity_info(chain_indices=[count_chains], sequence='', description='', entity_type=entity_type)
                    encoder.set_chain_info(chain_id=next(chain_id_iterator), chain_name='\x00' if len(chain.get_id().strip()) == 0 else chain.get_id(), num_groups=0)
                    if count_chains > 0:
                        groups_per_chain.append(count_groups - sum(groups_per_chain) - 1)
                    if not first_chain:
                        seqs.append(seq)
                    first_chain = False
                    count_chains += 1
                    seq = ''
                if entity_type == 'polymer':
                    seq += seq1(resname, custom_map=protein_letters_3to1_extended)
                prev_residue_type = residue_type
                prev_resname = resname
                encoder.set_group_info(group_name=resname, group_number=residue.id[1], insertion_code='\x00' if residue.id[2] == ' ' else residue.id[2], group_type='', atom_count=sum((1 for a in residue.get_unpacked_list() if select.accept_atom(a))), bond_count=0, single_letter_code=seq1(resname, custom_map=protein_letters_3to1_extended), sequence_index=len(seq) - 1 if entity_type == 'polymer' else -1, secondary_structure_type=-1)
                for atom in residue.get_unpacked_list():
                    if select.accept_atom(atom):
                        count_atoms += 1
                        encoder.set_atom_info(atom_name=atom.name, serial_number=count_atoms if renumber_atoms else atom.serial_number, alternative_location_id='\x00' if atom.altloc == ' ' else atom.altloc, x=atom.coord[0], y=atom.coord[1], z=atom.coord[2], occupancy=atom.occupancy, temperature_factor=atom.bfactor, element=atom.element, charge=0)
            seqs.append(seq)
            start_ind = len(encoder.entity_list) - len(seqs)
            for i, seq in enumerate(seqs):
                encoder.entity_list[start_ind + i]['sequence'] = seq
        chains_per_model.append(count_chains - sum(chains_per_model))
    groups_per_chain.append(count_groups - sum(groups_per_chain))
    encoder.chains_per_model = chains_per_model
    encoder.groups_per_chain = groups_per_chain
    encoder.num_atoms = count_atoms
    encoder.num_groups = count_groups
    encoder.num_chains = count_chains
    encoder.num_models = count_models
    encoder.finalize_structure()
    encoder.write_file(filepath)