from panel.pane import Perspective
def test_perspective_can_drag_to_filter():
    msg = {'filters': [['Curve', '==', None, 'integer', 'sum']]}
    actual = Perspective()._process_property_change(msg)
    assert actual == msg