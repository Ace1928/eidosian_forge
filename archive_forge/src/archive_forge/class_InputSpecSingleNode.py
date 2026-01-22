import sys
import os
import pytest
from nipype.pipeline import engine as pe
from nipype.interfaces import base as nib
class InputSpecSingleNode(nib.TraitedSpec):
    input1 = nib.traits.Int(desc='a random int')
    input2 = nib.traits.Int(desc='a random int')