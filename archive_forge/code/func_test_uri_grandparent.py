from pyxnat import uriutil
def test_uri_grandparent():
    uri = uriutil.uri_grandparent('/projects/1/subjects/2')
    assert uri == '/projects/1'