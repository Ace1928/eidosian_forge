import random
from .. import graphs
from . import links, twist
from spherogram.planarmap import random_map as raw_random_map
class LinkGenerationError(Exception):

    def __init__(self, num_tries):
        message = "Didn't generate requested link after %d tries." % num_tries
        Exception.__init__(self, message)