import os
import pytest
import rpy2.robjects as robjects
import rpy2.robjects.help as rh
def test_to_docstring(self):
    base_help = rh.Package('base')
    p = base_help.fetch('print')
    ds = p.to_docstring()
    assert ds[:5] == 'title'