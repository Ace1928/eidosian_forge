import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def write_magres(fd, image):
    """
    A writing function for magres files. Two steps: first data are arranged
    into structures, then dumped to the actual file
    """
    image_data = {}
    image_data['atoms'] = {'units': []}
    if np.all(image.get_pbc()):
        image_data['atoms']['units'].append(['lattice', 'Angstrom'])
        image_data['atoms']['lattice'] = [image.get_cell()]
    if image.has('labels'):
        labels = image.get_array('labels')
    else:
        labels = image.get_chemical_symbols()
    if image.has('indices'):
        indices = image.get_array('indices')
    else:
        indices = [labels[:i + 1].count(labels[i]) for i in range(len(labels))]
    symbols = image.get_array('castep_custom_species') if image.has('castep_custom_species') else image.get_chemical_symbols()
    atom_info = list(zip(symbols, image.get_positions()))
    if len(atom_info) > 0:
        image_data['atoms']['units'].append(['atom', 'Angstrom'])
        image_data['atoms']['atom'] = []
    for i, a in enumerate(atom_info):
        image_data['atoms']['atom'].append({'index': indices[i], 'position': a[1], 'species': a[0], 'label': labels[i]})
    if 'spacegroup' in image.info:
        image_data['atoms']['symmetry'] = [image.info['spacegroup'].symbol.replace(' ', '')]
    if 'magres_units' in image.info:
        image_data['magres'] = {'units': []}
        for u in image.info['magres_units']:
            p = u.split('_')[0]
            if p in _mprops:
                image_data['magres']['units'].append([u, image.info['magres_units'][u]])
                image_data['magres'][u] = []
                mn, order = _mprops[p]
                if order == 0:
                    tens = {mn: image.calc.results[u]}
                    image_data['magres'][u] = tens
                else:
                    arr = image.get_array(u)
                    li_tab = zip(labels, indices)
                    for i, (lab, ind) in enumerate(li_tab):
                        if order == 2:
                            for j, (lab2, ind2) in enumerate(li_tab[:i + 1]):
                                if arr[i][j] is not None:
                                    tens = {mn: arr[i][j], 'atom1': {'label': lab, 'index': ind}, 'atom2': {'label': lab2, 'index': ind2}}
                                    image_data['magres'][u].append(tens)
                        elif order == 1:
                            if arr[i] is not None:
                                tens = {mn: arr[i], 'atom': {'label': lab, 'index': ind}}
                                image_data['magres'][u].append(tens)
    if 'magresblock_calculation' in image.info:
        image_data['calculation'] = image.info['magresblock_calculation']

    def write_units(data, out):
        if 'units' in data:
            for tag, units in data['units']:
                out.append('  units %s %s' % (tag, units))

    def write_magres_block(data):
        """
            Write out a <magres> block from its dictionary representation
        """
        out = []

        def nout(tag, tensor_name):
            if tag in data:
                out.append(' '.join([' ', tag, tensor_string(data[tag][tensor_name])]))

        def siout(tag, tensor_name):
            if tag in data:
                for atom_si in data[tag]:
                    out.append('  %s %s %d %s' % (tag, atom_si['atom']['label'], atom_si['atom']['index'], tensor_string(atom_si[tensor_name])))
        write_units(data, out)
        nout('sus', 'S')
        siout('ms', 'sigma')
        siout('efg_local', 'V')
        siout('efg_nonlocal', 'V')
        siout('efg', 'V')

        def sisiout(tag, tensor_name):
            if tag in data:
                for isc in data[tag]:
                    out.append('  %s %s %d %s %d %s' % (tag, isc['atom1']['label'], isc['atom1']['index'], isc['atom2']['label'], isc['atom2']['index'], tensor_string(isc[tensor_name])))
        sisiout('isc_fc', 'K')
        sisiout('isc_orbital_p', 'K')
        sisiout('isc_orbital_d', 'K')
        sisiout('isc_spin', 'K')
        sisiout('isc', 'K')
        return '\n'.join(out)

    def write_atoms_block(data):
        out = []
        write_units(data, out)
        if 'lattice' in data:
            for lat in data['lattice']:
                out.append('  lattice %s' % tensor_string(lat))
        if 'symmetry' in data:
            for sym in data['symmetry']:
                out.append('  symmetry %s' % sym)
        if 'atom' in data:
            for a in data['atom']:
                out.append('  atom %s %s %s %s' % (a['species'], a['label'], a['index'], ' '.join((str(x) for x in a['position']))))
        return '\n'.join(out)

    def write_generic_block(data):
        out = []
        for tag, data in data.items():
            for value in data:
                out.append('%s %s' % (tag, ' '.join((str(x) for x in value))))
        return '\n'.join(out)
    block_writers = OrderedDict([('calculation', write_generic_block), ('atoms', write_atoms_block), ('magres', write_magres_block)])
    fd.write('#$magres-abinitio-v1.0\n')
    fd.write('# Generated by the Atomic Simulation Environment library\n')
    for b in block_writers:
        if b in image_data:
            fd.write('[{0}]\n'.format(b))
            fd.write(block_writers[b](image_data[b]))
            fd.write('\n[/{0}]\n'.format(b))
    for i in image.info:
        if '_' in i:
            ismag, b = i.split('_', 1)
            if ismag == 'magresblock' and b not in block_writers:
                fd.write('[{0}]\n'.format(b))
                fd.write(image.info[i])
                fd.write('[/{0}]\n'.format(b))