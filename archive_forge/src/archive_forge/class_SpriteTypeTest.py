import unittest
import pygame
from pygame import sprite
class SpriteTypeTest(SpriteBase, unittest.TestCase):
    Sprite = sprite.Sprite
    Groups = [sprite.Group, sprite.LayeredUpdates, sprite.RenderUpdates, sprite.OrderedUpdates]