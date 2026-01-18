import unittest
import pygame
from pygame import sprite
def test_switch_layer(self):
    sprites1 = []
    sprites2 = []
    layers = [3, 2, 3, 2, 3, 3, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 2, 3]
    for lay in layers:
        spr = self.sprite()
        spr._layer = lay
        self.LG.add(spr)
        if lay == 2:
            sprites1.append(spr)
        else:
            sprites2.append(spr)
    sprites1.sort(key=id)
    sprites2.sort(key=id)
    layer2_sprites = sorted(self.LG.get_sprites_from_layer(2), key=id)
    layer3_sprites = sorted(self.LG.get_sprites_from_layer(3), key=id)
    self.assertListEqual(sprites1, layer2_sprites)
    self.assertListEqual(sprites2, layer3_sprites)
    self.assertEqual(len(self.LG), len(sprites1) + len(sprites2))
    self.LG.switch_layer(2, 3)
    layer2_sprites = sorted(self.LG.get_sprites_from_layer(2), key=id)
    layer3_sprites = sorted(self.LG.get_sprites_from_layer(3), key=id)
    self.assertListEqual(sprites1, layer3_sprites)
    self.assertListEqual(sprites2, layer2_sprites)
    self.assertEqual(len(self.LG), len(sprites1) + len(sprites2))