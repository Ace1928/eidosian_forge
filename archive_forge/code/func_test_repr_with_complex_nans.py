from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,output', [([2, complex('nan'), 1], [' 2.0+0.0j', ' NaN+0.0j', ' 1.0+0.0j']), ([2, complex('nan'), -1], [' 2.0+0.0j', ' NaN+0.0j', '-1.0+0.0j']), ([-2, complex('nan'), -1], ['-2.0+0.0j', ' NaN+0.0j', '-1.0+0.0j']), ([-1.23j, complex('nan'), -1], ['-0.00-1.23j', '  NaN+0.00j', '-1.00+0.00j']), ([1.23j, complex('nan'), 1.23], [' 0.00+1.23j', '  NaN+0.00j', ' 1.23+0.00j']), ([-1.23j, complex(np.nan, np.nan), 1], ['-0.00-1.23j', '  NaN+ NaNj', ' 1.00+0.00j']), ([-1.23j, complex(1.2, np.nan), 1], ['-0.00-1.23j', ' 1.20+ NaNj', ' 1.00+0.00j']), ([-1.23j, complex(np.nan, -1.2), 1], ['-0.00-1.23j', '  NaN-1.20j', ' 1.00+0.00j'])])
@pytest.mark.parametrize('as_frame', [True, False])
def test_repr_with_complex_nans(data, output, as_frame):
    obj = Series(np.array(data))
    if as_frame:
        obj = obj.to_frame(name='val')
        reprs = [f'{i} {val}' for i, val in enumerate(output)]
        expected = f'{'val': >{len(reprs[0])}}\n' + '\n'.join(reprs)
    else:
        reprs = [f'{i}   {val}' for i, val in enumerate(output)]
        expected = '\n'.join(reprs) + '\ndtype: complex128'
    assert str(obj) == expected, f'\n{str(obj)}\n\n{expected}'