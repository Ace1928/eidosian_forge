from sympy.external.gmpy import MPZ, HAS_GMPY
from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.ring import Ring
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
import math
Compute factorial of ``a``. 