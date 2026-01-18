from panel.pane import Perspective
def test_perspective_can_filter_to_value():
    msg = {'filters': [['Curve', '==', 4]]}
    actual = Perspective()._process_property_change(msg)
    assert actual == msg