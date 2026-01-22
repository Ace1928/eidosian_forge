from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.colorpicker import ColorWheel, ColorPicker
class ColorPickerTest(GraphicUnitTest):

    def test_render(self):
        color_picker = ColorPicker()
        self.render(color_picker)

    def test_set_colour(self):
        color_picker = ColorPicker()
        self.assertEqual(color_picker.color, [1, 1, 1, 1])
        color_picker.set_color((0.5, 0.6, 0.7))
        self.assertEqual(color_picker.color, [0.5, 0.6, 0.7, 1])
        color_picker.set_color((0.5, 0.6, 0.7, 0.8))
        self.assertEqual(color_picker.color, [0.5, 0.6, 0.7, 0.8])