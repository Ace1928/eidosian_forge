import io
import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('values, vmin, vmax', [('positive', 0.5, 4.5), ('negative', -4.5, -0.5), ('mixed', -4.5, 4.5)])
@pytest.mark.parametrize('nullify', [None, 'vmin', 'vmax'])
@pytest.mark.parametrize('align', ['left', 'right', 'zero', 'mid'])
def test_vmin_vmax_widening(df_pos, df_neg, df_mix, values, vmin, vmax, nullify, align):
    if align == 'mid':
        if values == 'positive':
            align = 'left'
        elif values == 'negative':
            align = 'right'
    df = {'positive': df_pos, 'negative': df_neg, 'mixed': df_mix}[values]
    vmin = None if nullify == 'vmin' else vmin
    vmax = None if nullify == 'vmax' else vmax
    expand_df = df.copy()
    expand_df.loc[3, :], expand_df.loc[4, :] = (vmin, vmax)
    result = df.style.bar(align=align, vmin=vmin, vmax=vmax, color=['red', 'green'])._compute().ctx
    expected = expand_df.style.bar(align=align, color=['red', 'green'])._compute().ctx
    assert result.items() <= expected.items()