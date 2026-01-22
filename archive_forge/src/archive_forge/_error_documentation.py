import json
from collections import namedtuple
import macaroonbakery.bakery as bakery
Override the __new__ method so that we can
        have optional arguments, which namedtuple doesn't
        allow