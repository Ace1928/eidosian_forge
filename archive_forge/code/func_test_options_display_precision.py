import geopandas
import pytest
def test_options_display_precision():
    assert geopandas.options.display_precision is None
    geopandas.options.display_precision = 5
    assert geopandas.options.display_precision == 5
    with pytest.raises(ValueError):
        geopandas.options.display_precision = 'abc'
    with pytest.raises(ValueError):
        geopandas.options.display_precision = -1
    geopandas.options.display_precision = None