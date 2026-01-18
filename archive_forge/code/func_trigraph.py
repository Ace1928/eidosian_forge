import sys
import re
import copy
import time
import os.path
def trigraph(input):
    return _trigraph_pat.sub(lambda g: _trigraph_rep[g.group()[-1]], input)