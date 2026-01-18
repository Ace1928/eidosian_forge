import unittest
import pygame
from pygame import sprite
def test_repaint_rect_with_clip(self):
    group = self.LG
    surface = pygame.Surface((100, 100))
    group.set_clip(pygame.Rect(0, 0, 100, 100))
    group.repaint_rect(pygame.Rect(0, 0, 100, 100))
    group.draw(surface)