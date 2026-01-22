import os
import os.path as op
import datetime
import string
import networkx as nx
from ...utils.filemanip import split_filename
from ..base import (
from .base import CFFBaseInterface, have_cfflib
class CFFConverterOutputSpec(TraitedSpec):
    connectome_file = File(exists=True, desc='Output connectome file')