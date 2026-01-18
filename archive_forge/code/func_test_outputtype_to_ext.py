import os
import nipype.interfaces.fsl as fsl
from nipype.interfaces.base import InterfaceResult
from nipype.interfaces.fsl import check_fsl, no_fsl
import pytest
def test_outputtype_to_ext():
    for ftype, ext in fsl.Info.ftypes.items():
        res = fsl.Info.output_type_to_ext(ftype)
        assert res == ext
    with pytest.raises(KeyError):
        fsl.Info.output_type_to_ext('JUNK')