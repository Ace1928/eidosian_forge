import unittest
import pygame
from pygame import sprite
def test_sprites(self):
    sprites = []
    sprite_and_layer_count = 10
    for i in range(sprite_and_layer_count, 0, -1):
        sprites.append(self.sprite())
        sprites[-1]._layer = i
    self.LG.add(sprites)
    self.assertEqual(len(self.LG._spritelist), sprite_and_layer_count)
    expected_sprites = list(reversed(sprites))
    actual_sprites = self.LG.sprites()
    self.assertListEqual(actual_sprites, expected_sprites)