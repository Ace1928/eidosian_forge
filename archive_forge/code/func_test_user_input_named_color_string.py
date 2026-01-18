import pytest
from pandas import Series
from pandas.plotting._matplotlib.style import get_standard_colors
@pytest.mark.parametrize('color, num_colors, expected', [('Crimson', 1, ['Crimson']), ('DodgerBlue', 2, ['DodgerBlue', 'DodgerBlue']), ('firebrick', 3, ['firebrick', 'firebrick', 'firebrick'])])
def test_user_input_named_color_string(self, color, num_colors, expected):
    result = get_standard_colors(color=color, num_colors=num_colors)
    assert result == expected