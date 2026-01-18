from ase.atoms import Atoms
from ase.utils import reader, writer
@writer
def write_gen(fileobj, images):
    """Write structure in GEN format (refer to DFTB+ manual).
       Multiple snapshots are not allowed. """
    if not isinstance(images, (list, tuple)):
        images = [images]
    if len(images) != 1:
        raise ValueError('images contains more than one structure\n' + 'GEN format supports only single snapshot output')
    symbols = images[0].get_chemical_symbols()
    symboldict = dict()
    for sym in symbols:
        if not sym in symboldict:
            symboldict[sym] = len(symboldict) + 1
    orderedsymbols = list(['null'] * len(symboldict.keys()))
    for sym in symboldict.keys():
        orderedsymbols[symboldict[sym] - 1] = sym
    if images[0].pbc.any():
        pb_flag = 'S'
    else:
        pb_flag = 'C'
    natoms = len(symbols)
    ind = 0
    for atoms in images:
        fileobj.write('%d  %-5s\n' % (natoms, pb_flag))
        for s in orderedsymbols:
            fileobj.write('%-5s' % s)
        fileobj.write('\n')
        for sym, (x, y, z) in zip(symbols, atoms.get_positions()):
            ind += 1
            symbolid = symboldict[sym]
            fileobj.write('%-6d %d %22.15f %22.15f %22.15f\n' % (ind, symbolid, x, y, z))
    if images[0].pbc.any():
        fileobj.write('%22.15f %22.15f %22.15f \n' % (0.0, 0.0, 0.0))
        fileobj.write('%22.15f %22.15f %22.15f \n' % (images[0].get_cell()[0][0], images[0].get_cell()[0][1], images[0].get_cell()[0][2]))
        fileobj.write('%22.15f %22.15f %22.15f \n' % (images[0].get_cell()[1][0], images[0].get_cell()[1][1], images[0].get_cell()[1][2]))
        fileobj.write('%22.15f %22.15f %22.15f \n' % (images[0].get_cell()[2][0], images[0].get_cell()[2][1], images[0].get_cell()[2][2]))