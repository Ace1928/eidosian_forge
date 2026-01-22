import unittest
import os
import platform
import warnings
import pygame
class MouseModuleInteractiveTest(MouseTests):
    __tags__ = ['interactive']

    def test_set_pos(self):
        """Ensures set_pos works correctly.
        Requires tester to move the mouse to be on the window.
        """
        pygame.display.set_mode((500, 500))
        pygame.event.get()
        if not pygame.mouse.get_focused():
            return
        clock = pygame.time.Clock()
        expected_pos = ((10, 0), (0, 0), (499, 0), (499, 499), (341, 143), (94, 49))
        for x, y in expected_pos:
            pygame.mouse.set_pos(x, y)
            pygame.event.get()
            found_pos = pygame.mouse.get_pos()
            clock.tick()
            time_passed = 0.0
            ready_to_test = False
            while not ready_to_test and time_passed <= 1000.0:
                time_passed += clock.tick()
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEMOTION:
                        ready_to_test = True
            self.assertEqual(found_pos, (x, y))