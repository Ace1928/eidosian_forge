
import unittest
from main import SimulationGUI
from PyQt5.QtWidgets import QApplication
import sys

class TestSimulationGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)

    def setUp(self):
        self.gui = SimulationGUI()

    def test_create_particles(self):
        particles = self.gui.create_particles()
        self.assertGreater(len(particles), 0)

    def test_restart_simulation(self):
        self.gui.restart_simulation()
        self.assertGreater(len(self.gui.particles), 0)

    def test_toggle_pause(self):
        self.gui.toggle_pause()
        self.assertTrue(self.gui.is_paused)
        self.gui.toggle_pause()
        self.assertFalse(self.gui.is_paused)

    def test_update_simulation(self):
        self.gui.update_simulation()
        # Ensure no exceptions are raised during simulation update

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
