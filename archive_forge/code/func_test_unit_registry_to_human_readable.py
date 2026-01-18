from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_unit_registry_to_human_readable():
    d = defaultdict(lambda: 1)
    assert unit_registry_to_human_readable(d) == dict(((x, (1, 1)) for x in SI_base_registry.keys()))
    ur = {'length': 1000.0 * u.metre, 'mass': 0.01 * u.kilogram, 'time': 10000.0 * u.second, 'current': 0.1 * u.ampere, 'temperature': 10.0 * u.kelvin, 'luminous_intensity': 0.001 * u.candela, 'amount': 10000.0 * u.mole}
    assert unit_registry_to_human_readable(ur) == {'length': (1000.0, 'm'), 'mass': (0.01, 'kg'), 'time': (10000.0, 's'), 'current': (0.1, 'A'), 'temperature': (10.0, 'K'), 'luminous_intensity': (0.001, 'cd'), 'amount': (10000.0, 'mol')}
    assert unit_registry_to_human_readable(ur) != {'length': (100.0, 'm'), 'mass': (0.01, 'kg'), 'time': (10000.0, 's'), 'current': (0.1, 'A'), 'temperature': (10.0, 'K'), 'luminous_intensity': (0.001, 'cd'), 'amount': (10000.0, 'mol')}