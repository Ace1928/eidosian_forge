from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.simplify.simplify import simplify
from sympy.core.backend import (Matrix, sympify, Mul, Derivative, sin, cos,
def _validate_coordinates(coordinates=None, speeds=None, check_duplicates=True, is_dynamicsymbols=True):
    t_set = {dynamicsymbols._t}
    if coordinates is None:
        coordinates = []
    elif not iterable(coordinates):
        coordinates = [coordinates]
    if speeds is None:
        speeds = []
    elif not iterable(speeds):
        speeds = [speeds]
    if check_duplicates:
        seen = set()
        coord_duplicates = {x for x in coordinates if x in seen or seen.add(x)}
        seen = set()
        speed_duplicates = {x for x in speeds if x in seen or seen.add(x)}
        overlap = set(coordinates).intersection(speeds)
        if coord_duplicates:
            raise ValueError(f'The generalized coordinates {coord_duplicates} are duplicated, all generalized coordinates should be unique.')
        if speed_duplicates:
            raise ValueError(f'The generalized speeds {speed_duplicates} are duplicated, all generalized speeds should be unique.')
        if overlap:
            raise ValueError(f'{overlap} are defined as both generalized coordinates and generalized speeds.')
    if is_dynamicsymbols:
        for coordinate in coordinates:
            if not (isinstance(coordinate, (AppliedUndef, Derivative)) and coordinate.free_symbols == t_set):
                raise ValueError(f'Generalized coordinate "{coordinate}" is not a dynamicsymbol.')
        for speed in speeds:
            if not (isinstance(speed, (AppliedUndef, Derivative)) and speed.free_symbols == t_set):
                raise ValueError(f'Generalized speed "{speed}" is not a dynamicsymbol.')