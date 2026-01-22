from suds import *
from suds.sax import *
@classmethod
def __valid(cls, *args):
    return len(args) and args[0] is not None