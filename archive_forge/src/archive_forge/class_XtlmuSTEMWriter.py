import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.data import chemical_symbols
from ase.utils import reader, writer
from .utils import verify_cell_for_export, verify_dictionary
class XtlmuSTEMWriter:
    """See the docstring of the `write_mustem` function.
    """

    def __init__(self, atoms, keV, debye_waller_factors=None, comment=None, occupancies=None, fit_cell_to_atoms=False):
        verify_cell_for_export(atoms.get_cell())
        self.atoms = atoms.copy()
        self.atom_types = sorted(set(atoms.symbols))
        self.keV = keV
        self.comment = comment
        self.occupancies = self._get_occupancies(occupancies)
        self.RMS = self._get_RMS(debye_waller_factors)
        self.numbers = symbols2numbers(self.atom_types)
        if fit_cell_to_atoms:
            self.atoms.translate(-self.atoms.positions.min(axis=0))
            self.atoms.set_cell(self.atoms.positions.max(axis=0))

    def _get_occupancies(self, occupancies):
        if occupancies is None:
            if 'occupancies' in self.atoms.arrays:
                occupancies = {element: self._parse_array_from_atoms('occupancies', element, True) for element in self.atom_types}
            else:
                occupancies = 1.0
        if np.isscalar(occupancies):
            occupancies = {atom: occupancies for atom in self.atom_types}
        elif isinstance(occupancies, dict):
            verify_dictionary(self.atoms, occupancies, 'occupancies')
        return occupancies

    def _get_RMS(self, DW):
        if DW is None:
            if 'debye_waller_factors' in self.atoms.arrays:
                DW = {element: self._parse_array_from_atoms('debye_waller_factors', element, True) / (8 * np.pi ** 2) for element in self.atom_types}
        elif np.isscalar(DW):
            if len(self.atom_types) > 1:
                raise ValueError('This cell contains more then one type of atoms and the Debye-Waller factor needs to be provided for each atom using a dictionary.')
            DW = {self.atom_types[0]: DW / (8 * np.pi ** 2)}
        elif isinstance(DW, dict):
            verify_dictionary(self.atoms, DW, 'debye_waller_factors')
            for key, value in DW.items():
                DW[key] = value / (8 * np.pi ** 2)
        if DW is None:
            raise ValueError('Missing Debye-Waller factors. It can be provided as a dictionary with symbols as key or if the cell contains only a single type of element, the Debye-Waller factor can also be provided as float.')
        return DW

    def _parse_array_from_atoms(self, name, element, check_same_value):
        """
        Return the array "name" for the given element.

        Parameters
        ----------
        name : str
            The name of the arrays. Can be any key of `atoms.arrays`
        element : str, int
            The element to be considered.
        check_same_value : bool
            Check if all values are the same in the array. Necessary for
            'occupancies' and 'debye_waller_factors' arrays.

        Returns
        -------
        array containing the values corresponding defined by "name" for the
        given element. If check_same_value, return a single element.

        """
        if isinstance(element, str):
            element = symbols2numbers(element)[0]
        sliced_array = self.atoms.arrays[name][self.atoms.numbers == element]
        if check_same_value:
            if np.unique(sliced_array).size > 1:
                raise ValueError("All the '{}' values for element '{}' must be equal.".format(name, chemical_symbols[element]))
            sliced_array = sliced_array[0]
        return sliced_array

    def _get_position_array_single_atom_type(self, number):
        return self.atoms.get_scaled_positions()[self.atoms.numbers == number]

    def _get_file_header(self):
        if self.comment is None:
            s = '{0} atoms with chemical formula: {1}\n'.format(len(self.atoms), self.atoms.get_chemical_formula())
        else:
            s = self.comment
        s += '{} {} {} {} {} {}\n'.format(*self.atoms.cell.cellpar().tolist())
        s += '{}\n'.format(self.keV)
        s += '{}\n'.format(len(self.atom_types))
        return s

    def _get_element_header(self, atom_type, number, atom_type_number, occupancy, RMS):
        return '{0}\n{1} {2} {3} {4:.3g}\n'.format(atom_type, number, atom_type_number, occupancy, RMS)

    def _get_file_end(self):
        return 'Orientation\n   1 0 0\n   0 1 0\n   0 0 1\n'

    def write_to_file(self, fd):
        if isinstance(fd, str):
            fd = open(fd, 'w')
        fd.write(self._get_file_header())
        for atom_type, number, occupancy in zip(self.atom_types, self.numbers, self.occupancies):
            positions = self._get_position_array_single_atom_type(number)
            atom_type_number = positions.shape[0]
            fd.write(self._get_element_header(atom_type, atom_type_number, number, self.occupancies[atom_type], self.RMS[atom_type]))
            np.savetxt(fname=fd, X=positions, fmt='%.6g', newline='\n')
        fd.write(self._get_file_end())