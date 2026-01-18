import unittest
import pygame
from pygame import sprite
def test_remove_internal(self):
    for g in self.groups:
        self.sprite.add_internal(g)
    for g in self.groups:
        self.sprite.remove_internal(g)
    for g in self.groups:
        self.assertFalse(g in self.sprite.groups())