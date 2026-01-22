import gast as ast
import os
import re
from time import time
class AnalysisContext(object):
    """
    Class that stores the hierarchy of node visited.

    Contains:
        * parent module
        * parent function
    """

    def __init__(self):
        self.module = None
        self.function = None