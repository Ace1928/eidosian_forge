import unittest
import pygame
from pygame import sprite
def _nondirty_intersections_redrawn(self, use_source_rect=False):
    RED = pygame.Color('red')
    BLUE = pygame.Color('blue')
    WHITE = pygame.Color('white')
    YELLOW = pygame.Color('yellow')
    surface = pygame.Surface((60, 80))
    surface.fill(WHITE)
    start_pos = (10, 10)
    red_sprite_source = pygame.Rect((45, 0), (5, 4))
    blue_sprite_source = pygame.Rect((0, 40), (20, 10))
    image_source = pygame.Surface((50, 50))
    image_source.fill(YELLOW)
    image_source.fill(RED, red_sprite_source)
    image_source.fill(BLUE, blue_sprite_source)
    blue_sprite = pygame.sprite.DirtySprite(self.LG)
    if use_source_rect:
        blue_sprite.image = image_source
        blue_sprite.rect = pygame.Rect(start_pos, (blue_sprite_source.w - 7, blue_sprite_source.h - 7))
        blue_sprite.source_rect = blue_sprite_source
        start_x, start_y = blue_sprite.rect.topleft
        end_x = start_x + blue_sprite.source_rect.w
        end_y = start_y + blue_sprite.source_rect.h
    else:
        blue_sprite.image = image_source.subsurface(blue_sprite_source)
        blue_sprite.rect = pygame.Rect(start_pos, blue_sprite_source.size)
        start_x, start_y = blue_sprite.rect.topleft
        end_x, end_y = blue_sprite.rect.bottomright
    red_sprite = pygame.sprite.DirtySprite(self.LG)
    red_sprite.image = image_source
    red_sprite.rect = pygame.Rect(start_pos, red_sprite_source.size)
    red_sprite.source_rect = red_sprite_source
    red_sprite.dirty = 2
    for _ in range(4):
        red_sprite.rect.move_ip(2, 1)
        self.LG.draw(surface)
    surface.lock()
    try:
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if red_sprite.rect.collidepoint(x, y):
                    expected_color = RED
                else:
                    expected_color = BLUE
                color = surface.get_at((x, y))
                self.assertEqual(color, expected_color, f'pos=({x}, {y})')
    finally:
        surface.unlock()