import pytest
from pandas import Series
from pandas.plotting._matplotlib.style import get_standard_colors
@pytest.mark.parametrize('num_colors, expected', [(1, ['r', 'g', 'b', 'k']), (2, ['r', 'g', 'b', 'k']), (3, ['r', 'g', 'b', 'k']), (4, ['r', 'g', 'b', 'k']), (5, ['r', 'g', 'b', 'k', 'r']), (6, ['r', 'g', 'b', 'k', 'r', 'g'])])
def test_user_input_color_string(self, num_colors, expected):
    color = 'rgbk'
    result = get_standard_colors(color=color, num_colors=num_colors)
    assert result == expected