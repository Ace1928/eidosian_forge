from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.utils import writer
@writer
def write_x3d(fd, atoms, format='X3D'):
    """Writes to html using X3DOM.

    Args:
        filename - str or file-like object, filename or output file object
        atoms - Atoms object to be rendered
        format - str, either 'X3DOM' for web-browser compatibility or 'X3D'
            to be readable by Blender. `None` to detect format based on file
            extension ('.html' -> 'X3DOM', '.x3d' -> 'X3D')"""
    X3D(atoms).write(fd, datatype=format)