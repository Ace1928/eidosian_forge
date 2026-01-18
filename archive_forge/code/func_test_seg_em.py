import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from .. import EM
@pytest.mark.skipif(no_nifty_tool(cmd='seg_EM'), reason='niftyseg is not installed')
def test_seg_em():
    seg_em = EM()
    cmd = get_custom_path('seg_EM', env_dir='NIFTYSEGDIR')
    assert seg_em.cmd == cmd
    with pytest.raises(ValueError):
        seg_em.run()
    in_file = example_data('im1.nii')
    seg_em.inputs.in_file = in_file
    seg_em.inputs.no_prior = 4
    cmd_tmp = '{cmd} -in {in_file} -nopriors 4 -bc_out {bc_out} -out {out_file} -out_outlier {out_outlier}'
    expected_cmd = cmd_tmp.format(cmd=cmd, in_file=in_file, out_file='im1_em.nii.gz', bc_out='im1_bc_em.nii.gz', out_outlier='im1_outlier_em.nii.gz')
    assert seg_em.cmdline == expected_cmd