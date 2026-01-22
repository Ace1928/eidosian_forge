from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
def _fill_coordinate_list(self, coordinates, n_coords, label='q', offset=0, number_single=False):
    """Helper method for _generate_coordinates and _generate_speeds.

        Parameters
        ==========

        coordinates : iterable
            Iterable of coordinates or speeds that have been provided.
        n_coords : Integer
            Number of coordinates that should be returned.
        label : String, optional
            Coordinate type either 'q' (coordinates) or 'u' (speeds). The
            Default is 'q'.
        offset : Integer
            Count offset when creating new dynamicsymbols. The default is 0.
        number_single : Boolean
            Boolean whether if n_coords == 1, number should still be used. The
            default is False.

        """

    def create_symbol(number):
        if n_coords == 1 and (not number_single):
            return dynamicsymbols(f'{label}_{self.name}')
        return dynamicsymbols(f'{label}{number}_{self.name}')
    name = 'generalized coordinate' if label == 'q' else 'generalized speed'
    generated_coordinates = []
    if coordinates is None:
        coordinates = []
    elif not iterable(coordinates):
        coordinates = [coordinates]
    if not (len(coordinates) == 0 or len(coordinates) == n_coords):
        raise ValueError(f'Expected {n_coords} {name}s, instead got {len(coordinates)} {name}s.')
    for i, coord in enumerate(coordinates):
        if coord is None:
            generated_coordinates.append(create_symbol(i + offset))
        elif isinstance(coord, (AppliedUndef, Derivative)):
            generated_coordinates.append(coord)
        else:
            raise TypeError(f'The {name} {coord} should have been a dynamicsymbol.')
    for i in range(len(coordinates) + offset, n_coords + offset):
        generated_coordinates.append(create_symbol(i))
    return Matrix(generated_coordinates)