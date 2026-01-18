from panel.layout import Spacer
def test_spacer_clone():
    spacer = Spacer(width=400, height=300)
    clone = spacer.clone()
    assert {k: v for k, v in spacer.param.values().items() if k != 'name'} == {k: v for k, v in clone.param.values().items() if k != 'name'}