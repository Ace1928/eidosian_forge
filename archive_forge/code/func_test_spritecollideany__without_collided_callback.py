import unittest
import pygame
from pygame import sprite
def test_spritecollideany__without_collided_callback(self):
    expected_sprite = self.s2
    collided_sprite = sprite.spritecollideany(self.s1, self.ag2)
    self.assertEqual(collided_sprite, expected_sprite)
    self.s2.rect.move_ip(0, 10)
    collided_sprite = sprite.spritecollideany(self.s1, self.ag2)
    self.assertIsNone(collided_sprite)
    self.s3.rect.move_ip(-105, -105)
    expected_sprite = self.s3
    collided_sprite = sprite.spritecollideany(self.s1, self.ag2)
    self.assertEqual(collided_sprite, expected_sprite)
    self.s2.rect.move_ip(0, -10)
    expected_sprite_choices = self.ag2.sprites()
    collided_sprite = sprite.spritecollideany(self.s1, self.ag2)
    self.assertIn(collided_sprite, expected_sprite_choices)