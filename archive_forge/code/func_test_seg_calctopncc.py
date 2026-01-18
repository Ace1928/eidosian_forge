import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from .. import LabelFusion, CalcTopNCC
@pytest.mark.skipif(no_nifty_tool(cmd='seg_CalcTopNCC'), reason='niftyseg is not installed')
def test_seg_calctopncc():
    """Test interfaces for seg_CalctoNCC"""
    calctopncc = CalcTopNCC()
    cmd = get_custom_path('seg_CalcTopNCC', env_dir='NIFTYSEGDIR')
    assert calctopncc.cmd == cmd
    with pytest.raises(ValueError):
        calctopncc.run()
    in_file = example_data('im1.nii')
    file1 = example_data('im2.nii')
    file2 = example_data('im3.nii')
    calctopncc.inputs.in_file = in_file
    calctopncc.inputs.num_templates = 2
    calctopncc.inputs.in_templates = [file1, file2]
    calctopncc.inputs.top_templates = 1
    cmd_tmp = '{cmd} -target {in_file} -templates 2 {file1} {file2} -n 1'
    expected_cmd = cmd_tmp.format(cmd=cmd, in_file=in_file, file1=file1, file2=file2)
    assert calctopncc.cmdline == expected_cmd