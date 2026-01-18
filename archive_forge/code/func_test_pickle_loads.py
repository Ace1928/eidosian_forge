from .._pickle_api import pickle_dumps, pickle_loads
def test_pickle_loads():
    msg = pickle_dumps({'hello': 'world', 'test': 123})
    data = pickle_loads(msg)
    assert len(data) == 2
    assert data['hello'] == 'world'
    assert data['test'] == 123