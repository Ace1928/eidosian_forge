import gast as ast
import os
import re
from time import time
class NodeAnalysis(Analysis):
    """An analysis that operates on any node."""