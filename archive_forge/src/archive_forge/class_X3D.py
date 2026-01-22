from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.utils import writer
class X3D:
    """Class to write either X3D (readable by open-source rendering
    programs such as Blender) or X3DOM html, readable by modern web
    browsers.
    """

    def __init__(self, atoms):
        self._atoms = atoms

    def write(self, fileobj, datatype):
        """Writes output to either an 'X3D' or an 'X3DOM' file, based on
        the extension. For X3D, filename should end in '.x3d'. For X3DOM,
        filename should end in '.html'.

        Args:
            datatype - str, output format. 'X3D' or 'X3DOM'.
        """
        w = WriteToFile(fileobj)
        if datatype == 'X3DOM':
            w(0, '<html>')
            w(1, '<head>')
            w(2, '<title>ASE atomic visualization</title>')
            w(2, '<link rel="stylesheet" type="text/css"')
            w(2, ' href="https://www.x3dom.org/x3dom/release/x3dom.css">')
            w(2, '</link>')
            w(2, '<script type="text/javascript"')
            w(2, ' src="https://www.x3dom.org/x3dom/release/x3dom.js">')
            w(2, '</script>')
            w(1, '</head>')
            w(1, '<body>')
            w(2, '<X3D>')
        elif datatype == 'X3D':
            w(0, '<?xml version="1.0" encoding="UTF-8"?>')
            w(0, '<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.2//EN" "http://www.web3d.org/specifications/x3d-3.2.dtd">')
            w(0, '<X3D profile="Interchange" version="3.2" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.2.xsd">')
        else:
            raise ValueError('datatype not supported: ' + str(datatype))
        w(3, '<Scene>')
        for atom in self._atoms:
            for indent, line in atom_lines(atom):
                w(4 + indent, line)
        w(3, '</Scene>')
        if datatype == 'X3DOM':
            w(2, '</X3D>')
            w(1, '</body>')
            w(0, '</html>')
        elif datatype == 'X3D':
            w(0, '</X3D>')