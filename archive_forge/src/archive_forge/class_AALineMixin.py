import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
class AALineMixin(BaseLineMixin):
    """Mixin test for drawing a single aaline.

    This class contains all the general single aaline drawing tests.
    """

    def test_aaline__args(self):
        """Ensures draw aaline accepts the correct args."""
        bounds_rect = self.draw_aaline(pygame.Surface((3, 3)), (0, 10, 0, 50), (0, 0), (1, 1), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__args_without_blend(self):
        """Ensures draw aaline accepts the args without a blend."""
        bounds_rect = self.draw_aaline(pygame.Surface((2, 2)), (0, 0, 0, 50), (0, 0), (2, 2))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__blend_warning(self):
        """From pygame 2, blend=False should raise DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.draw_aaline(pygame.Surface((2, 2)), (0, 0, 0, 50), (0, 0), (2, 2), False)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_aaline__kwargs(self):
        """Ensures draw aaline accepts the correct kwargs"""
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        start_pos = (1, 1)
        end_pos = (2, 2)
        kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_aaline(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__kwargs_order_independent(self):
        """Ensures draw aaline's kwargs are not order dependent."""
        bounds_rect = self.draw_aaline(start_pos=(1, 2), end_pos=(2, 1), color=(10, 20, 30), surface=pygame.Surface((3, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__args_missing(self):
        """Ensures draw aaline detects any missing required args."""
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface, color, (0, 0))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline()

    def test_aaline__kwargs_missing(self):
        """Ensures draw aaline detects any missing required kwargs."""
        kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'start_pos': (2, 1), 'end_pos': (2, 2)}
        for name in ('end_pos', 'start_pos', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**invalid_kwargs)

    def test_aaline__arg_invalid_types(self):
        """Ensures draw aaline detects invalid arg types."""
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        start_pos = (0, 1)
        end_pos = (1, 2)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface, color, start_pos, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface, color, (1,), end_pos)
        with self.assertRaises(ValueError):
            bounds_rect = self.draw_aaline(surface, 'invalid-color', start_pos, end_pos)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline((1, 2, 3, 4), color, start_pos, end_pos)

    def test_aaline__kwarg_invalid_types(self):
        """Ensures draw aaline detects invalid kwarg types."""
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        start_pos = (1, 0)
        end_pos = (2, 0)
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}, {'surface': surface, 'color': 2.3, 'start_pos': start_pos, 'end_pos': end_pos}, {'surface': surface, 'color': color, 'start_pos': (0, 0, 0), 'end_pos': end_pos}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': (0,)}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__kwarg_invalid_name(self):
        """Ensures draw aaline detects invalid kwarg names."""
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        start_pos = (1, 1)
        end_pos = (2, 0)
        kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'invalid': 1}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__args_and_kwargs(self):
        """Ensures draw aaline accepts a combination of args/kwargs"""
        surface = pygame.Surface((3, 2))
        color = (255, 255, 0, 0)
        start_pos = (0, 1)
        end_pos = (1, 2)
        kwargs = {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}
        for name in ('surface', 'color', 'start_pos', 'end_pos'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_aaline(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_aaline(surface, color, **kwargs)
            elif 'start_pos' == name:
                bounds_rect = self.draw_aaline(surface, color, start_pos, **kwargs)
            elif 'end_pos' == name:
                bounds_rect = self.draw_aaline(surface, color, start_pos, end_pos, **kwargs)
            else:
                bounds_rect = self.draw_aaline(surface, color, start_pos, end_pos, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__valid_start_pos_formats(self):
        """Ensures draw aaline accepts different start_pos formats."""
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'start_pos': None, 'end_pos': (2, 2)}
        x, y = (2, 1)
        positions = ((x, y), (x + 0.01, y), (x, y + 0.01), (x + 0.01, y + 0.01))
        for start_pos in positions:
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['start_pos'] = seq_type(start_pos)
                bounds_rect = self.draw_aaline(**kwargs)
                color = surface.get_at((x, y))
                for i, sub_color in enumerate(expected_color):
                    self.assertGreaterEqual(color[i] + 6, sub_color, start_pos)
                self.assertIsInstance(bounds_rect, pygame.Rect, start_pos)

    def test_aaline__valid_end_pos_formats(self):
        """Ensures draw aaline accepts different end_pos formats."""
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'start_pos': (2, 1), 'end_pos': None}
        x, y = (2, 2)
        positions = ((x, y), (x + 0.02, y), (x, y + 0.02), (x + 0.02, y + 0.02))
        for end_pos in positions:
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['end_pos'] = seq_type(end_pos)
                bounds_rect = self.draw_aaline(**kwargs)
                color = surface.get_at((x, y))
                for i, sub_color in enumerate(expected_color):
                    self.assertGreaterEqual(color[i] + 15, sub_color, end_pos)
                self.assertIsInstance(bounds_rect, pygame.Rect, end_pos)

    def test_aaline__invalid_start_pos_formats(self):
        """Ensures draw aaline handles invalid start_pos formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': None, 'end_pos': (2, 2)}
        start_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
        for start_pos in start_pos_fmts:
            kwargs['start_pos'] = start_pos
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__invalid_end_pos_formats(self):
        """Ensures draw aaline handles invalid end_pos formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': (2, 2), 'end_pos': None}
        end_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
        for end_pos in end_pos_fmts:
            kwargs['end_pos'] = end_pos
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__valid_color_formats(self):
        """Ensures draw aaline accepts different color formats."""
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': None, 'start_pos': pos, 'end_pos': (2, 1)}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_aaline(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__invalid_color_formats(self):
        """Ensures draw aaline handles invalid color formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'start_pos': (1, 1), 'end_pos': (2, 1)}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__color(self):
        """Tests if the aaline drawn is the correct color."""
        pos = (0, 0)
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_aaline(surface, expected_color, pos, (1, 0))
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_aaline__gaps(self):
        """Tests if the aaline drawn contains any gaps.

        See: #512
        """
        expected_color = (255, 255, 255)
        for surface in self._create_surfaces():
            width = surface.get_width()
            self.draw_aaline(surface, expected_color, (0, 0), (width - 1, 0))
            for x in range(width):
                pos = (x, 0)
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_aaline__bounding_rect(self):
        """Ensures draw aaline returns the correct bounding rect.

        Tests lines with endpoints on and off the surface.
        """
        line_color = pygame.Color('red')
        surf_color = pygame.Color('blue')
        width = height = 30
        helper_rect = pygame.Rect((0, 0), (width, height))
        for size in ((width + 5, height + 5), (width - 5, height - 5)):
            surface = pygame.Surface(size, 0, 32)
            surf_rect = surface.get_rect()
            for pos in rect_corners_mids_and_center(surf_rect):
                helper_rect.center = pos
                for start, end in self._rect_lines(helper_rect):
                    surface.fill(surf_color)
                    bounding_rect = self.draw_aaline(surface, line_color, start, end)
                    expected_rect = create_bounding_rect(surface, surf_color, start)
                    self.assertEqual(bounding_rect, expected_rect)

    def test_aaline__surface_clip(self):
        """Ensures draw aaline respects a surface's clip area."""
        surfw = surfh = 30
        aaline_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for center in rect_corners_mids_and_center(clip_rect):
            pos_rect.center = center
            surface.set_clip(None)
            surface.fill(surface_color)
            self.draw_aaline(surface, aaline_color, pos_rect.midtop, pos_rect.midbottom)
            expected_pts = get_color_points(surface, surface_color, clip_rect, False)
            surface.fill(surface_color)
            surface.set_clip(clip_rect)
            self.draw_aaline(surface, aaline_color, pos_rect.midtop, pos_rect.midbottom)
            surface.lock()
            for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                if pt in expected_pts:
                    self.assertNotEqual(surface.get_at(pt), surface_color, pt)
                else:
                    self.assertEqual(surface.get_at(pt), surface_color, pt)
            surface.unlock()