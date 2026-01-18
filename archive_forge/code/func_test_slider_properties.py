from kivy.tests.common import GraphicUnitTest
def test_slider_properties(self):
    from kivy.uix.slider import Slider
    r = self.render
    r(Slider(value=25))
    r(Slider(value=50))
    r(Slider(value=100))
    r(Slider(min=-100, max=100, value=0))
    r(Slider(orientation='vertical', value=25))
    r(Slider(orientation='vertical', value=50))
    r(Slider(orientation='vertical', value=100))
    r(Slider(orientation='vertical', min=-100, max=100, value=0))