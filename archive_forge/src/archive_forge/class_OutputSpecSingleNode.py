import sys
import os
import pytest
from nipype.pipeline import engine as pe
from nipype.interfaces import base as nib
class OutputSpecSingleNode(nib.TraitedSpec):
    output1 = nib.traits.Int(desc='a random int')