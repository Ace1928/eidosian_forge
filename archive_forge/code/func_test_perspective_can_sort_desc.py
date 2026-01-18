from panel.pane import Perspective
def test_perspective_can_sort_desc():
    msg = {'sort': [['Curve', 'desc']]}
    actual = Perspective()._process_property_change(msg)
    assert actual == msg