import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from .. import UnaryStats, BinaryStats
@pytest.mark.skipif(no_nifty_tool(cmd='seg_stats'), reason='niftyseg is not installed')
def test_unary_stats():
    """Test for the seg_stats interfaces"""
    unarys = UnaryStats()
    cmd = get_custom_path('seg_stats', env_dir='NIFTYSEGDIR')
    assert unarys.cmd == cmd
    with pytest.raises(ValueError):
        unarys.run()
    in_file = example_data('im1.nii')
    unarys.inputs.in_file = in_file
    unarys.inputs.operation = 'a'
    expected_cmd = '{cmd} {in_file} -a'.format(cmd=cmd, in_file=in_file)
    assert unarys.cmdline == expected_cmd