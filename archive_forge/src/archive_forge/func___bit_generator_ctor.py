from .mtrand import RandomState
from ._philox import Philox
from ._pcg64 import PCG64, PCG64DXSM
from ._sfc64 import SFC64
from ._generator import Generator
from ._mt19937 import MT19937
def __bit_generator_ctor(bit_generator_name='MT19937'):
    """
    Pickling helper function that returns a bit generator object

    Parameters
    ----------
    bit_generator_name : str
        String containing the name of the BitGenerator

    Returns
    -------
    bit_generator : BitGenerator
        BitGenerator instance
    """
    if bit_generator_name in BitGenerators:
        bit_generator = BitGenerators[bit_generator_name]
    else:
        raise ValueError(str(bit_generator_name) + ' is not a known BitGenerator module.')
    return bit_generator()