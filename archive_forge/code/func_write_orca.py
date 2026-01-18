from io import StringIO
from ase.io import read
from ase.utils import reader
def write_orca(atoms, **params):
    """Function to write ORCA input file
    """
    charge = params['charge']
    mult = params['mult']
    label = params['label']
    if 'pcpot' in params.keys():
        pcpot = params['pcpot']
        pcstring = '% pointcharges "' + label + '.pc"\n\n'
        params['orcablocks'] += pcstring
        pcpot.write_mmcharges(label)
    with open(label + '.inp', 'w') as fd:
        fd.write('! engrad %s \n' % params['orcasimpleinput'])
        fd.write('%s \n' % params['orcablocks'])
        fd.write('*xyz')
        fd.write(' %d' % charge)
        fd.write(' %d \n' % mult)
        for atom in atoms:
            if atom.tag == 71:
                symbol = atom.symbol + ' : '
            else:
                symbol = atom.symbol + '   '
            fd.write(symbol + str(atom.position[0]) + ' ' + str(atom.position[1]) + ' ' + str(atom.position[2]) + '\n')
        fd.write('*\n')