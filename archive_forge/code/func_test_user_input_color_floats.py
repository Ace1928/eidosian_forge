import pytest
from pandas import Series
from pandas.plotting._matplotlib.style import get_standard_colors
@pytest.mark.parametrize('num_colors, expected', [(1, [(0.1, 0.2, 0.3)]), (2, [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]), (3, [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)])])
def test_user_input_color_floats(self, num_colors, expected):
    color = (0.1, 0.2, 0.3)
    result = get_standard_colors(color=color, num_colors=num_colors)
    assert result == expected