import unittest
import pygame
from pygame import sprite
class AbstractGroupTypeTest(unittest.TestCase):

    def setUp(self):
        self.ag = sprite.AbstractGroup()
        self.ag2 = sprite.AbstractGroup()
        self.s1 = sprite.Sprite(self.ag)
        self.s2 = sprite.Sprite(self.ag)
        self.s3 = sprite.Sprite(self.ag2)
        self.s4 = sprite.Sprite(self.ag2)
        self.s1.image = pygame.Surface((10, 10))
        self.s1.image.fill(pygame.Color('red'))
        self.s1.rect = self.s1.image.get_rect()
        self.s2.image = pygame.Surface((10, 10))
        self.s2.image.fill(pygame.Color('green'))
        self.s2.rect = self.s2.image.get_rect()
        self.s2.rect.left = 10
        self.s3.image = pygame.Surface((10, 10))
        self.s3.image.fill(pygame.Color('blue'))
        self.s3.rect = self.s3.image.get_rect()
        self.s3.rect.top = 10
        self.s4.image = pygame.Surface((10, 10))
        self.s4.image.fill(pygame.Color('white'))
        self.s4.rect = self.s4.image.get_rect()
        self.s4.rect.left = 10
        self.s4.rect.top = 10
        self.bg = pygame.Surface((20, 20))
        self.scr = pygame.Surface((20, 20))
        self.scr.fill(pygame.Color('grey'))

    def test_has(self):
        """See if AbstractGroup.has() works as expected."""
        self.assertEqual(True, self.s1 in self.ag)
        self.assertEqual(True, self.ag.has(self.s1))
        self.assertEqual(True, self.ag.has([self.s1, self.s2]))
        self.assertNotEqual(True, self.ag.has([self.s1, self.s2, self.s3]))
        self.assertNotEqual(True, self.ag.has(self.s1, self.s2, self.s3))
        self.assertNotEqual(True, self.ag.has(self.s1, sprite.Group(self.s2, self.s3)))
        self.assertNotEqual(True, self.ag.has(self.s1, [self.s2, self.s3]))
        self.assertFalse(self.ag.has(*[]))
        self.assertFalse(self.ag.has([]))
        self.assertFalse(self.ag.has([[]]))
        self.assertEqual(True, self.ag2.has(self.s3))

    def test_add(self):
        ag3 = sprite.AbstractGroup()
        sprites = (self.s1, self.s2, self.s3, self.s4)
        for s in sprites:
            self.assertNotIn(s, ag3)
        ag3.add(self.s1, [self.s2], self.ag2)
        for s in sprites:
            self.assertIn(s, ag3)

    def test_add_internal(self):
        self.assertNotIn(self.s1, self.ag2)
        self.ag2.add_internal(self.s1)
        self.assertIn(self.s1, self.ag2)

    def test_clear(self):
        self.ag.draw(self.scr)
        self.ag.clear(self.scr, self.bg)
        self.assertEqual((0, 0, 0, 255), self.scr.get_at((5, 5)))
        self.assertEqual((0, 0, 0, 255), self.scr.get_at((15, 5)))

    def test_draw(self):
        self.ag.draw(self.scr)
        self.assertEqual((255, 0, 0, 255), self.scr.get_at((5, 5)))
        self.assertEqual((0, 255, 0, 255), self.scr.get_at((15, 5)))
        self.assertEqual(self.ag.spritedict[self.s1], pygame.Rect(0, 0, 10, 10))
        self.assertEqual(self.ag.spritedict[self.s2], pygame.Rect(10, 0, 10, 10))

    def test_empty(self):
        self.ag.empty()
        self.assertFalse(self.s1 in self.ag)
        self.assertFalse(self.s2 in self.ag)

    def test_has_internal(self):
        self.assertTrue(self.ag.has_internal(self.s1))
        self.assertFalse(self.ag.has_internal(self.s3))

    def test_remove(self):
        self.ag.remove(self.s1)
        self.assertFalse(self.ag in self.s1.groups())
        self.assertFalse(self.ag.has(self.s1))
        self.ag2.remove(self.s3, self.s4)
        self.assertFalse(self.ag2 in self.s3.groups())
        self.assertFalse(self.ag2 in self.s4.groups())
        self.assertFalse(self.ag2.has(self.s3, self.s4))
        self.ag.add(self.s1, self.s3, self.s4)
        self.ag2.add(self.s3, self.s4)
        g = sprite.Group(self.s2)
        self.ag.remove([self.s1, g], self.ag2)
        self.assertFalse(self.ag in self.s1.groups())
        self.assertFalse(self.ag in self.s2.groups())
        self.assertFalse(self.ag in self.s3.groups())
        self.assertFalse(self.ag in self.s4.groups())
        self.assertFalse(self.ag.has(self.s1, self.s2, self.s3, self.s4))

    def test_remove_internal(self):
        self.ag.remove_internal(self.s1)
        self.assertFalse(self.ag.has_internal(self.s1))

    def test_sprites(self):
        expected_sprites = sorted((self.s1, self.s2), key=id)
        sprite_list = sorted(self.ag.sprites(), key=id)
        self.assertListEqual(sprite_list, expected_sprites)

    def test_update(self):

        class test_sprite(pygame.sprite.Sprite):
            sink = []

            def __init__(self, *groups):
                pygame.sprite.Sprite.__init__(self, *groups)

            def update(self, *args):
                self.sink += args
        s = test_sprite(self.ag)
        self.ag.update(1, 2, 3)
        self.assertEqual(test_sprite.sink, [1, 2, 3])

    def test_update_with_kwargs(self):

        class test_sprite(pygame.sprite.Sprite):
            sink = []
            sink_kwargs = {}

            def __init__(self, *groups):
                pygame.sprite.Sprite.__init__(self, *groups)

            def update(self, *args, **kwargs):
                self.sink += args
                self.sink_kwargs.update(kwargs)
        s = test_sprite(self.ag)
        self.ag.update(1, 2, 3, foo=4, bar=5)
        self.assertEqual(test_sprite.sink, [1, 2, 3])
        self.assertEqual(test_sprite.sink_kwargs, {'foo': 4, 'bar': 5})