import os
import pytest
from ....utils.filemanip import which
from ....testing import example_data
from .. import (
@pytest.mark.skipif(no_nifty_tool(cmd='reg_measure'), reason='niftyreg is not installed. reg_measure not found.')
def test_reg_measure():
    """tests for reg_measure interface"""
    nr_measure = RegMeasure()
    assert nr_measure.cmd == get_custom_path('reg_measure')
    with pytest.raises(ValueError):
        nr_measure.run()
    ref_file = example_data('im1.nii')
    flo_file = example_data('im2.nii')
    nr_measure.inputs.ref_file = ref_file
    nr_measure.inputs.flo_file = flo_file
    nr_measure.inputs.measure_type = 'lncc'
    nr_measure.inputs.omp_core_val = 4
    cmd_tmp = '{cmd} -flo {flo} -lncc -omp 4 -out {out} -ref {ref}'
    expected_cmd = cmd_tmp.format(cmd=get_custom_path('reg_measure'), flo=flo_file, out='im2_lncc.txt', ref=ref_file)
    assert nr_measure.cmdline == expected_cmd