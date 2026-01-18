from pyxnat import select
def test_simple_level_expand():
    assert select.compute('/projects/IMAGEN//experiments') == ['/project/IMAGEN/subjects/*/experiments/*']