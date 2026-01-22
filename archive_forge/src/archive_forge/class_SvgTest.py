from kivy.tests.common import GraphicUnitTest
class SvgTest(GraphicUnitTest):

    def test_simple(self):
        import xml.etree.ElementTree as ET
        from kivy.uix.widget import Widget
        from kivy.graphics.svg import Svg
        wid = Widget()
        with wid.canvas:
            svg = Svg()
            svg.set_tree(ET.ElementTree(ET.fromstring(SIMPLE_SVG)))
        self.render(wid)

    def test_scale(self):
        import xml.etree.ElementTree as ET
        from kivy.uix.widget import Widget
        from kivy.graphics.svg import Svg
        wid = Widget()
        with wid.canvas:
            svg = Svg()
            svg.set_tree(ET.ElementTree(ET.fromstring(SCALE_SVG)))
        self.render(wid)

    def test_rotate(self):
        import xml.etree.ElementTree as ET
        from kivy.uix.widget import Widget
        from kivy.graphics.svg import Svg
        wid = Widget()
        with wid.canvas:
            svg = Svg()
            svg.set_tree(ET.ElementTree(ET.fromstring(ROTATE_SVG)))
        self.render(wid)