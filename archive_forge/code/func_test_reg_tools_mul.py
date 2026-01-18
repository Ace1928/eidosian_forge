import os
import pytest
from ....utils.filemanip import which
from ....testing import example_data
from .. import (
@pytest.mark.skipif(no_nifty_tool(cmd='reg_tools'), reason='niftyreg is not installed. reg_tools not found.')
def test_reg_tools_mul():
    """tests for reg_tools interface"""
    nr_tools = RegTools()
    assert nr_tools.cmd == get_custom_path('reg_tools')
    with pytest.raises(ValueError):
        nr_tools.run()
    in_file = example_data('im1.nii')
    nr_tools.inputs.in_file = in_file
    nr_tools.inputs.mul_val = 4
    nr_tools.inputs.omp_core_val = 4
    cmd_tmp = '{cmd} -in {in_file} -mul 4.0 -omp 4 -out {out_file}'
    expected_cmd = cmd_tmp.format(cmd=get_custom_path('reg_tools'), in_file=in_file, out_file='im1_tools.nii.gz')
    assert nr_tools.cmdline == expected_cmd
    nr_tools_2 = RegTools(iso_flag=True, omp_core_val=4)
    in_file = example_data('im1.nii')
    nr_tools_2.inputs.in_file = in_file
    cmd_tmp = '{cmd} -in {in_file} -iso -omp 4 -out {out_file}'
    expected_cmd = cmd_tmp.format(cmd=get_custom_path('reg_tools'), in_file=in_file, out_file='im1_tools.nii.gz')
    assert nr_tools_2.cmdline == expected_cmd