import os
from monty.re import regrep
def test_regrep():
    """
    We are making sure a file containing line numbers is read in reverse
    order, i.e. the first line that is read corresponds to the last line.
    number
    """
    fname = os.path.join(test_dir, '3000_lines.txt')
    matches = regrep(fname, {'1': '1(\\d+)', '3': '3(\\d+)'}, postprocess=int)
    assert len(matches['1']) == 1380
    assert len(matches['3']) == 571
    assert matches['1'][0][0][0] == 0
    matches = regrep(fname, {'1': '1(\\d+)', '3': '3(\\d+)'}, reverse=True, terminate_on_match=True, postprocess=int)
    assert len(matches['1']) == 1
    assert len(matches['3']) == 11