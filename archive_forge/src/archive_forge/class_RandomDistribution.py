import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class RandomDistribution(NumberGenerator, TimeAwareRandomState):
    """
    The base class for all Numbergenerators using random state.

    Numbergen provides a hierarchy of classes to make it easier to use
    the random distributions made available in Python's random module,
    where each class is tied to a particular random distribution.

    RandomDistributions support setting parameters on creation rather
    than passing them each call, and allow pickling to work properly.
    Code that uses these classes will be independent of how many
    parameters are used by the underlying distribution, and can simply
    treat them as a generic source of random numbers.

    RandomDistributions are examples of TimeAwareRandomState, and thus
    can be locked to a global time if desired.  By default,
    time_dependent=False, and so a new random value will be generated
    each time these objects are called.  If you have a global time
    function, you can set time_dependent=True, so that the random
    values will instead be constant at any given time, changing only
    when the time changes.  Using time_dependent values can help you
    obtain fully reproducible streams of random numbers, even if you
    e.g. move time forwards and backwards for testing.

    Note: Each RandomDistribution object has independent random state.
    """
    seed = param.Integer(default=None, allow_None=True, doc='\n       Sets the seed of the random number generator and can be used to\n       randomize time dependent streams.\n\n       If seed is None, there is no control over the random stream\n       (i.e. no reproducibility of the stream).')
    __abstract = True

    def __init__(self, **params):
        """
        Initialize a new Random() instance and store the supplied
        positional and keyword arguments.

        If seed=X is specified, sets the Random() instance's seed.
        Otherwise, calls creates an unseeded Random instance which is
        likely to result in a state very different from any just used.
        """
        super().__init__(**params)
        self._initialize_random_state(seed=self.seed, shared=False)

    def __call__(self):
        if self.time_dependent:
            self._hash_and_seed()