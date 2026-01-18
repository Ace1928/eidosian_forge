import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def word_wrap(surf, text, font, margin=0, color=(0, 0, 0)):
    font.origin = True
    surf_width, surf_height = surf.get_size()
    width = surf_width - 2 * margin
    height = surf_height - 2 * margin
    line_spacing = int(1.25 * font.get_sized_height())
    x, y = (margin, margin + line_spacing)
    space = font.get_rect(' ')
    for word in iwords(text):
        if word == '\n':
            x, y = (margin, y + line_spacing)
        else:
            bounds = font.get_rect(word)
            if x + bounds.width + bounds.x >= width:
                x, y = (margin, y + line_spacing)
            if x + bounds.width + bounds.x >= width:
                raise ValueError('word too wide for the surface')
            if y + bounds.height - bounds.y >= height:
                raise ValueError('text to long for the surface')
            font.render_to(surf, (x, y), None, color)
            x += bounds.width + space.width
    return (x, y)