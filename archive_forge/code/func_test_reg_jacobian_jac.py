import os
import pytest
from ....utils.filemanip import which
from ....testing import example_data
from .. import (
@pytest.mark.skipif(no_nifty_tool(cmd='reg_jacobian'), reason='niftyreg is not installed. reg_jacobian not found.')
def test_reg_jacobian_jac():
    """Test interface for RegJacobian"""
    nr_jacobian = RegJacobian()
    assert nr_jacobian.cmd == get_custom_path('reg_jacobian')
    with pytest.raises(ValueError):
        nr_jacobian.run()
    ref_file = example_data('im1.nii')
    trans_file = example_data('warpfield.nii')
    nr_jacobian.inputs.ref_file = ref_file
    nr_jacobian.inputs.trans_file = trans_file
    nr_jacobian.inputs.omp_core_val = 4
    cmd_tmp = '{cmd} -omp 4 -ref {ref} -trans {trans} -jac {jac}'
    expected_cmd = cmd_tmp.format(cmd=get_custom_path('reg_jacobian'), ref=ref_file, trans=trans_file, jac='warpfield_jac.nii.gz')
    assert nr_jacobian.cmdline == expected_cmd
    nr_jacobian_2 = RegJacobian(type='jacM', omp_core_val=4)
    ref_file = example_data('im1.nii')
    trans_file = example_data('warpfield.nii')
    nr_jacobian_2.inputs.ref_file = ref_file
    nr_jacobian_2.inputs.trans_file = trans_file
    cmd_tmp = '{cmd} -omp 4 -ref {ref} -trans {trans} -jacM {jac}'
    expected_cmd = cmd_tmp.format(cmd=get_custom_path('reg_jacobian'), ref=ref_file, trans=trans_file, jac='warpfield_jacM.nii.gz')
    assert nr_jacobian_2.cmdline == expected_cmd
    nr_jacobian_3 = RegJacobian(type='jacL', omp_core_val=4)
    ref_file = example_data('im1.nii')
    trans_file = example_data('warpfield.nii')
    nr_jacobian_3.inputs.ref_file = ref_file
    nr_jacobian_3.inputs.trans_file = trans_file
    cmd_tmp = '{cmd} -omp 4 -ref {ref} -trans {trans} -jacL {jac}'
    expected_cmd = cmd_tmp.format(cmd=get_custom_path('reg_jacobian'), ref=ref_file, trans=trans_file, jac='warpfield_jacL.nii.gz')
    assert nr_jacobian_3.cmdline == expected_cmd