import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
class SumInputSpec(nib.TraitedSpec):
    input1 = nib.traits.List(nib.traits.Int, mandatory=True, desc='input')