import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase.io.xsd import SetChild, _write_xsd_html
from ase import Atoms
def write_xtd(filename, images, connectivity=None, moviespeed=10):
    """Takes Atoms object, and write materials studio file
    atoms: Atoms object
    filename: path of the output file
    moviespeed: speed of animation. between 0 and 10

    note: material studio file cannot use a partial periodic system. If partial
    perodic system was inputted, full periodicity was assumed.
    """
    if moviespeed < 0 or moviespeed > 10:
        raise ValueError('moviespeed only between 0 and 10 allowed')
    if hasattr(images, 'get_positions'):
        images = [images]
    XSD, ATR = _write_xsd_html(images, connectivity)
    ATR.attrib['NumChildren'] = '2'
    natoms = len(images[0])
    bonds = list()
    if connectivity is not None:
        for i in range(connectivity.shape[0]):
            for j in range(i + 1, connectivity.shape[0]):
                if connectivity[i, j]:
                    bonds.append([i, j])
    s = '!BIOSYM archive 3\n'
    if not images[0].pbc.all():
        SetChild(ATR, 'Trajectory', dict(ID=str(natoms + 3 + len(bonds)), Increment='-1', End=str(len(images)), Type='arc', Speed=str(moviespeed), FrameClassType='Atom'))
        s += 'PBC=OFF\n'
        for image in images:
            s += _image_header
            s += '\n'
            an = image.get_chemical_symbols()
            xyz = image.get_positions()
            for i in range(natoms):
                s += _get_atom_str(an[i], xyz[i, :])
            s += _image_footer
    else:
        SetChild(ATR, 'Trajectory', dict(ID=str(natoms + 9 + len(bonds)), Increment='-1', End=str(len(images)), Type='arc', Speed=str(moviespeed), FrameClassType='Atom'))
        s += 'PBC=ON\n'
        for image in images:
            s += _image_header
            s += 'PBC'
            vec = image.cell.lengths()
            s += '{:>10.4f}{:>10.4f}{:>10.4f}'.format(vec[0], vec[1], vec[2])
            angles = image.cell.angles()
            s += '{:>10.4f}{:>10.4f}{:>10.4f}'.format(*angles)
            s += '\n'
            an = image.get_chemical_symbols()
            angrad = np.deg2rad(angles)
            cell = np.zeros((3, 3))
            cell[0, :] = [vec[0], 0, 0]
            cell[1, :] = np.array([np.cos(angrad[2]), np.sin(angrad[2]), 0]) * vec[1]
            cell[2, 0] = vec[2] * np.cos(angrad[1])
            cell[2, 1] = (vec[1] * vec[2] * np.cos(angrad[0]) - cell[1, 0] * cell[2, 0]) / cell[1, 1]
            cell[2, 2] = np.sqrt(vec[2] ** 2 - cell[2, 0] ** 2 - cell[2, 1] ** 2)
            xyz = np.dot(image.get_scaled_positions(), cell)
            for i in range(natoms):
                s += _get_atom_str(an[i], xyz[i, :])
            s += _image_footer
    if isinstance(filename, str):
        farcname = filename[:-3] + 'arc'
    else:
        farcname = filename.name[:-3] + 'arc'
    with open(farcname, 'w') as farc:
        farc.write(s)
    openandclose = False
    try:
        if isinstance(filename, str):
            fd = open(filename, 'w')
            openandclose = True
        else:
            fd = filename
        rough_string = ET.tostring(XSD, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        Document = reparsed.toprettyxml(indent='\t')
        fd.write(Document)
    finally:
        if openandclose:
            fd.close()