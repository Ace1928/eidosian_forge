from panel.pane import Perspective
def test_perspective_int_cols(document, comm):
    psp = Perspective(data, columns=[0], aggregates={0: 'mean'}, sort=[[0, 'desc']], group_by=[0], split_by=[0], filters=[[0, '==', 'None']])
    model = psp.get_root(document, comm)
    assert '0' in model.source.data
    assert model.columns == ['0']
    assert model.group_by == ['0']
    assert model.split_by == ['0']
    assert model.aggregates == {'0': 'mean'}
    assert model.filters == [['0', '==', 'None']]
    assert model.sort == [['0', 'desc']]
    psp2 = Perspective(data)
    psp2._process_events({'columns': ['0'], 'group_by': ['0'], 'split_by': ['0'], 'aggregates': {'0': 'mean'}, 'filters': [['0', '==', 'None']], 'sort': [['0', 'desc']]})
    assert psp2.columns == [0]
    assert psp2.group_by == [0]
    assert psp2.split_by == [0]
    assert psp2.aggregates == {0: 'mean'}
    assert psp2.sort == [[0, 'desc']]
    assert psp2.filters == [[0, '==', 'None']]