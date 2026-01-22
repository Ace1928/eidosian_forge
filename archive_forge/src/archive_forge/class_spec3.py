import os
import warnings
import pytest
from ....utils.filemanip import split_filename
from ... import base as nib
from ...base import traits, Undefined
from ....interfaces import fsl
from ...utility.wrappers import Function
from ....pipeline import Node
from ..specs import get_filecopy_info
class spec3(nib.TraitedSpec):
    moo = nib.File(exists=True, name_source='doo')
    doo = nib.traits.List(nib.File(exists=True))