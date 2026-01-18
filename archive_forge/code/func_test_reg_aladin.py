import pytest
from ....testing import example_data
from .. import get_custom_path, RegAladin, RegF3D
from .test_regutils import no_nifty_tool
@pytest.mark.skipif(no_nifty_tool(cmd='reg_aladin'), reason='niftyreg is not installed. reg_aladin not found.')
def test_reg_aladin():
    """tests for reg_aladin interface"""
    nr_aladin = RegAladin()
    assert nr_aladin.cmd == get_custom_path('reg_aladin')
    with pytest.raises(ValueError):
        nr_aladin.run()
    ref_file = example_data('im1.nii')
    flo_file = example_data('im2.nii')
    rmask_file = example_data('mask.nii')
    nr_aladin.inputs.ref_file = ref_file
    nr_aladin.inputs.flo_file = flo_file
    nr_aladin.inputs.rmask_file = rmask_file
    nr_aladin.inputs.omp_core_val = 4
    cmd_tmp = '{cmd} -aff {aff} -flo {flo} -omp 4 -ref {ref} -res {res} -rmask {rmask}'
    expected_cmd = cmd_tmp.format(cmd=get_custom_path('reg_aladin'), aff='im2_aff.txt', flo=flo_file, ref=ref_file, res='im2_res.nii.gz', rmask=rmask_file)
    assert nr_aladin.cmdline == expected_cmd