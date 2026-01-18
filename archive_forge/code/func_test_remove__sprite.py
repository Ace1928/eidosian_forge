import unittest
import pygame
from pygame import sprite
def test_remove__sprite(self):
    sprites = []
    sprite_count = 10
    for i in range(sprite_count):
        sprites.append(self.sprite())
        sprites[-1].rect = pygame.Rect((0, 0, 0, 0))
    self.LG.add(sprites)
    self.assertEqual(len(self.LG._spritelist), sprite_count)
    for i in range(sprite_count):
        self.LG.remove(sprites[i])
    self.assertEqual(len(self.LG._spritelist), 0)