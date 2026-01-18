import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
@pytest.mark.parametrize('theme_name', ['theme_grey', 'theme_classic', 'theme_dark', 'theme_grey', 'theme_light', 'theme_bw', 'theme_linedraw', 'theme_void', 'theme_minimal'])
def test_theme(self, theme_name):
    theme = getattr(ggplot2, theme_name)
    gp = ggplot2.ggplot(mtcars) + theme()
    assert isinstance(gp, ggplot2.GGPlot)