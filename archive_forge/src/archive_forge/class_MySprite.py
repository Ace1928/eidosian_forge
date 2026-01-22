import unittest
import pygame
from pygame import sprite
class MySprite(sprite.Sprite):

    def __init__(self, *args, **kwargs):
        sprite.Sprite.__init__(self, *args, **kwargs)
        self.image = pygame.Surface((2, 4), 0, 24)
        self.rect = self.image.get_rect()