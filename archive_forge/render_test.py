
import unittest
from units import Unit
from particles import Particle
from render import Renderer

class TestRenderer(unittest.TestCase):
    def setUp(self):
        self.unit1 = Unit(5.0, 6.0, 7.0, 1.0, 10.0, 2.0, 3.0, 5.0, (1.0, 2.0, 3.0))
        self.unit2 = Unit(4.0, 5.0, 6.0, 1.1, 12.0, 3.0, 4.0, 6.0, (4.0, 5.0, 6.0))
        self.units = [self.unit1, self.unit2]
        self.particle = Particle(self.units)
        self.renderer = Renderer()

    def test_draw_unit(self):
        self.renderer.draw_unit(self.unit1, 'red')
        # Check if the unit is added to the plot
        self.assertEqual(len(self.renderer.ax.collections), 1)

    def test_draw_link(self):
        self.unit1.links.append((self.unit2, 'single'))
        self.renderer.draw_link(self.unit1, self.unit2, 'single')
        # Check if the link is added to the plot
        self.assertEqual(len(self.renderer.ax.lines), 1)

    def test_render_space(self):
        self.renderer.render_space([self.particle], 300.0, 0)
        # Ensure no exceptions were raised during rendering

    def test_get_color_based_on_property(self):
        color = self.renderer.get_color_based_on_property(self.unit1, 0)
        self.assertIsInstance(color, str)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
