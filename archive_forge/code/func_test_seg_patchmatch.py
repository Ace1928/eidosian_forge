import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from .. import PatchMatch
@pytest.mark.skipif(no_nifty_tool(cmd='seg_PatchMatch'), reason='niftyseg is not installed')
def test_seg_patchmatch():
    seg_patchmatch = PatchMatch()
    cmd = get_custom_path('seg_PatchMatch', env_dir='NIFTYSEGDIR')
    assert seg_patchmatch.cmd == cmd
    with pytest.raises(ValueError):
        seg_patchmatch.run()
    in_file = example_data('im1.nii')
    mask_file = example_data('im2.nii')
    db_file = example_data('db.xml')
    seg_patchmatch.inputs.in_file = in_file
    seg_patchmatch.inputs.mask_file = mask_file
    seg_patchmatch.inputs.database_file = db_file
    cmd_tmp = '{cmd} -i {in_file} -m {mask_file} -db {db} -o {out_file}'
    expected_cmd = cmd_tmp.format(cmd=cmd, in_file=in_file, mask_file=mask_file, db=db_file, out_file='im1_pm.nii.gz')
    assert seg_patchmatch.cmdline == expected_cmd