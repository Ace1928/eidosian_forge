from pyxnat import uriutil
def test_uri_last():
    assert uriutil.uri_last('/projects/1/subjects/2') == '2'