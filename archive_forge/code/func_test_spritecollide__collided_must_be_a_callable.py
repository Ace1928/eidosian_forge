import unittest
import pygame
from pygame import sprite
def test_spritecollide__collided_must_be_a_callable(self):
    self.assertRaises(TypeError, sprite.spritecollide, self.s1, self.ag2, dokill=False, collided=1)