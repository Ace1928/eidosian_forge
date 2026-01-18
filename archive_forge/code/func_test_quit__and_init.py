import sys
import unittest
import platform
import pygame
def test_quit__and_init(self):
    self.not_init_assertions()
    pygame.init()
    self.init_assertions()
    pygame.quit()
    self.not_init_assertions()