import unittest
import pygame
from pygame import sprite
class DirtyWeakSpriteTypeTest(DirtySpriteTypeTest, WeakSpriteTypeTest):
    Sprite = sprite.WeakDirtySprite