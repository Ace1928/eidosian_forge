import numpy as np
from ase import io
from ase import build
from ase.io.espresso import parse_position_line
from pytest import approx
def test_pw_input():
    """Read pw input file."""
    with open('pw_input.pwi', 'w') as pw_input_f:
        pw_input_f.write(pw_input_text)
    pw_input_atoms = io.read('pw_input.pwi', format='espresso-in')
    assert len(pw_input_atoms) == 8
    assert pw_input_atoms.get_initial_magnetic_moments() == approx([5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 0.0, 0.0])