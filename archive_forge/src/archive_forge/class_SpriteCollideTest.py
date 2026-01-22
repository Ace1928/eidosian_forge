import unittest
import pygame
from pygame import sprite
class SpriteCollideTest(unittest.TestCase):

    def setUp(self):
        self.ag = sprite.AbstractGroup()
        self.ag2 = sprite.AbstractGroup()
        self.s1 = sprite.Sprite(self.ag)
        self.s2 = sprite.Sprite(self.ag2)
        self.s3 = sprite.Sprite(self.ag2)
        self.s1.image = pygame.Surface((50, 10), pygame.SRCALPHA, 32)
        self.s2.image = pygame.Surface((10, 10), pygame.SRCALPHA, 32)
        self.s3.image = pygame.Surface((10, 10), pygame.SRCALPHA, 32)
        self.s1.rect = self.s1.image.get_rect()
        self.s2.rect = self.s2.image.get_rect()
        self.s3.rect = self.s3.image.get_rect()
        self.s2.rect.move_ip(40, 0)
        self.s3.rect.move_ip(100, 100)

    def test_spritecollide__works_if_collided_cb_is_None(self):
        self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=None), [self.s2])

    def test_spritecollide__works_if_collided_cb_not_passed(self):
        self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False), [self.s2])

    def test_spritecollide__collided_must_be_a_callable(self):
        self.assertRaises(TypeError, sprite.spritecollide, self.s1, self.ag2, dokill=False, collided=1)

    def test_spritecollide__collided_defaults_to_collide_rect(self):
        self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_rect), [self.s2])

    def test_collide_rect_ratio__ratio_of_one_like_default(self):
        self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_rect_ratio(1.0)), [self.s2])

    def test_collide_rect_ratio__collides_all_at_ratio_of_twenty(self):
        collided_func = sprite.collide_rect_ratio(20.0)
        expected_sprites = sorted(self.ag2.sprites(), key=id)
        collided_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
        self.assertListEqual(collided_sprites, expected_sprites)

    def test_collide_circle__no_radius_set(self):
        self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_circle), [self.s2])

    def test_collide_circle_ratio__no_radius_and_ratio_of_one(self):
        self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_circle_ratio(1.0)), [self.s2])

    def test_collide_circle_ratio__no_radius_and_ratio_of_twenty(self):
        collided_func = sprite.collide_circle_ratio(20.0)
        expected_sprites = sorted(self.ag2.sprites(), key=id)
        collided_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
        self.assertListEqual(expected_sprites, collided_sprites)

    def test_collide_circle__radius_set_by_collide_circle_ratio(self):
        collided_func = sprite.collide_circle_ratio(20.0)
        sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func)
        self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_circle), [self.s2])

    def test_collide_circle_ratio__no_radius_and_ratio_of_two_twice(self):
        collided_func = sprite.collide_circle_ratio(2.0)
        expected_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
        collided_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
        self.assertListEqual(expected_sprites, collided_sprites)

    def test_collide_circle__with_radii_set(self):
        self.s1.radius = 50
        self.s2.radius = 10
        self.s3.radius = 400
        collided_func = sprite.collide_circle
        expected_sprites = sorted(self.ag2.sprites(), key=id)
        collided_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
        self.assertListEqual(expected_sprites, collided_sprites)

    def test_collide_circle_ratio__with_radii_set(self):
        self.s1.radius = 50
        self.s2.radius = 10
        self.s3.radius = 400
        collided_func = sprite.collide_circle_ratio(0.5)
        expected_sprites = sorted(self.ag2.sprites(), key=id)
        collided_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
        self.assertListEqual(expected_sprites, collided_sprites)

    def test_collide_mask__opaque(self):
        self.s1.image.fill((255, 255, 255, 255))
        self.s2.image.fill((255, 255, 255, 255))
        self.s3.image.fill((255, 255, 255, 255))
        self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_mask), [self.s2])
        self.s1.mask = pygame.mask.from_surface(self.s1.image)
        self.s2.mask = pygame.mask.from_surface(self.s2.image)
        self.s3.mask = pygame.mask.from_surface(self.s3.image)
        self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_mask), [self.s2])

    def test_collide_mask__transparent(self):
        self.s1.image.fill((255, 255, 255, 0))
        self.s2.image.fill((255, 255, 255, 0))
        self.s3.image.fill((255, 255, 255, 0))
        self.s1.mask = pygame.mask.from_surface(self.s1.image, 255)
        self.s2.mask = pygame.mask.from_surface(self.s2.image, 255)
        self.s3.mask = pygame.mask.from_surface(self.s3.image, 255)
        self.assertFalse(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_mask))

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

    def test_spritecollideany__with_collided_callback(self):
        arg_dict_a = {}
        arg_dict_b = {}
        return_container = [True]

        def collided_callback(spr_a, spr_b, arg_dict_a=arg_dict_a, arg_dict_b=arg_dict_b, return_container=return_container):
            count = arg_dict_a.get(spr_a, 0)
            arg_dict_a[spr_a] = 1 + count
            count = arg_dict_b.get(spr_b, 0)
            arg_dict_b[spr_b] = 1 + count
            return return_container[0]
        expected_sprite_choices = self.ag2.sprites()
        collided_sprite = sprite.spritecollideany(self.s1, self.ag2, collided_callback)
        self.assertIn(collided_sprite, expected_sprite_choices)
        self.assertEqual(len(arg_dict_a), 1)
        self.assertEqual(arg_dict_a[self.s1], 1)
        self.assertEqual(len(arg_dict_b), 1)
        self.assertEqual(list(arg_dict_b.values())[0], 1)
        self.assertTrue(self.s2 in arg_dict_b or self.s3 in arg_dict_b)
        arg_dict_a.clear()
        arg_dict_b.clear()
        return_container[0] = False
        collided_sprite = sprite.spritecollideany(self.s1, self.ag2, collided_callback)
        self.assertIsNone(collided_sprite)
        self.assertEqual(len(arg_dict_a), 1)
        self.assertEqual(arg_dict_a[self.s1], len(self.ag2))
        self.assertEqual(len(arg_dict_b), len(self.ag2))
        for s in self.ag2:
            self.assertEqual(arg_dict_b[s], 1)

    def test_groupcollide__without_collided_callback(self):
        expected_dict = {self.s1: [self.s2]}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False)
        self.assertDictEqual(expected_dict, crashed)
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False)
        self.assertDictEqual(expected_dict, crashed)
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, True)
        self.assertDictEqual(expected_dict, crashed)
        expected_dict = {}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False)
        self.assertDictEqual(expected_dict, crashed)
        self.s3.rect.move_ip(-100, -100)
        expected_dict = {self.s1: [self.s3]}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, True, False)
        self.assertDictEqual(expected_dict, crashed)
        expected_dict = {}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False)
        self.assertDictEqual(expected_dict, crashed)

    def test_groupcollide__with_collided_callback(self):
        collided_callback_true = lambda spr_a, spr_b: True
        collided_callback_false = lambda spr_a, spr_b: False
        expected_dict = {}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False, collided_callback_false)
        self.assertDictEqual(expected_dict, crashed)
        expected_dict = {self.s1: sorted(self.ag2.sprites(), key=id)}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False, collided_callback_true)
        for value in crashed.values():
            value.sort(key=id)
        self.assertDictEqual(expected_dict, crashed)
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False, collided_callback_true)
        for value in crashed.values():
            value.sort(key=id)
        self.assertDictEqual(expected_dict, crashed)
        expected_dict = {}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, True, collided_callback_false)
        self.assertDictEqual(expected_dict, crashed)
        expected_dict = {self.s1: sorted(self.ag2.sprites(), key=id)}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, True, collided_callback_true)
        for value in crashed.values():
            value.sort(key=id)
        self.assertDictEqual(expected_dict, crashed)
        expected_dict = {}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, True, collided_callback_true)
        self.assertDictEqual(expected_dict, crashed)
        self.ag.add(self.s2)
        self.ag2.add(self.s3)
        expected_dict = {}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, True, False, collided_callback_false)
        self.assertDictEqual(expected_dict, crashed)
        expected_dict = {self.s1: [self.s3], self.s2: [self.s3]}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, True, False, collided_callback_true)
        self.assertDictEqual(expected_dict, crashed)
        expected_dict = {}
        crashed = pygame.sprite.groupcollide(self.ag, self.ag2, True, False, collided_callback_true)
        self.assertDictEqual(expected_dict, crashed)

    def test_collide_rect(self):
        self.assertTrue(pygame.sprite.collide_rect(self.s1, self.s2))
        self.assertTrue(pygame.sprite.collide_rect(self.s2, self.s1))
        self.s2.rect.center = self.s3.rect.center
        self.assertTrue(pygame.sprite.collide_rect(self.s2, self.s3))
        self.assertTrue(pygame.sprite.collide_rect(self.s3, self.s2))
        self.s2.rect.inflate_ip(10, 10)
        self.assertTrue(pygame.sprite.collide_rect(self.s2, self.s3))
        self.assertTrue(pygame.sprite.collide_rect(self.s3, self.s2))
        self.s2.rect.center = (self.s1.rect.right, self.s1.rect.bottom)
        self.assertTrue(pygame.sprite.collide_rect(self.s1, self.s2))
        self.assertTrue(pygame.sprite.collide_rect(self.s2, self.s1))
        self.assertFalse(pygame.sprite.collide_rect(self.s1, self.s3))
        self.assertFalse(pygame.sprite.collide_rect(self.s3, self.s1))