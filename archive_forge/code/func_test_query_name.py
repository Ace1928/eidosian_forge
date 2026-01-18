from urllib.error import URLError
import pytest
import xyzservices.providers as xyz
from xyzservices import Bunch, TileProvider
def test_query_name():
    options = ['CartoDB Positron', 'cartodbpositron', 'cartodb-positron', 'carto db/positron', 'CARTO_DB_POSITRON', 'CartoDB.Positron', 'Carto,db,positron']
    for option in options:
        queried = xyz.query_name(option)
        assert isinstance(queried, TileProvider)
        assert queried.name == 'CartoDB.Positron'
    with pytest.raises(ValueError, match='No matching provider found'):
        xyz.query_name("i don't exist")
    option_with_underscore = 'NASAGIBS.ASTER_GDEM_Greyscale_Shaded_Relief'
    queried = xyz.query_name(option_with_underscore)
    assert isinstance(queried, TileProvider)
    assert queried.name == option_with_underscore