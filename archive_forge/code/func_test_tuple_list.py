import pytest
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata, iter_fields
from urllib3.packages.six import b, u
def test_tuple_list(self):
    for fieldname, value in iter_fields([('a', 'b')]):
        assert (fieldname, value) == ('a', 'b')
    assert list(iter_fields([('a', 'b'), ('c', 'd')])) == [('a', 'b'), ('c', 'd')]