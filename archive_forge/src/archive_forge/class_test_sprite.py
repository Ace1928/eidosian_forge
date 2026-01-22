import unittest
import pygame
from pygame import sprite
class test_sprite(pygame.sprite.Sprite):
    sink = []
    sink_dict = {}

    def __init__(self, *groups):
        pygame.sprite.Sprite.__init__(self, *groups)

    def update(self, *args, **kwargs):
        self.sink += args
        self.sink_dict.update(kwargs)