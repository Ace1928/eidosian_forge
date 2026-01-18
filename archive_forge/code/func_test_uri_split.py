from pyxnat import uriutil
def test_uri_split():
    uri = uriutil.uri_split('/projects/1/subjects/2')
    assert uri == ['/projects/1/subjects', '2']