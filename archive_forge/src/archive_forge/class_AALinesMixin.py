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
class AALinesMixin(BaseLineMixin):
    """Mixin test for drawing aalines.

    This class contains all the general aalines drawing tests.
    """

    def test_aalines__args(self):
        """Ensures draw aalines accepts the correct args."""
        bounds_rect = self.draw_aalines(pygame.Surface((3, 3)), (0, 10, 0, 50), False, ((0, 0), (1, 1)), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__args_without_blend(self):
        """Ensures draw aalines accepts the args without a blend."""
        bounds_rect = self.draw_aalines(pygame.Surface((2, 2)), (0, 0, 0, 50), False, ((0, 0), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__blend_warning(self):
        """From pygame 2, blend=False should raise DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.draw_aalines(pygame.Surface((2, 2)), (0, 0, 0, 50), False, ((0, 0), (1, 1)), False)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_aalines__kwargs(self):
        """Ensures draw aalines accepts the correct kwargs."""
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        points = ((0, 0), (1, 1), (2, 2))
        kwargs_list = [{'surface': surface, 'color': color, 'closed': False, 'points': points}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_aalines(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__kwargs_order_independent(self):
        """Ensures draw aalines's kwargs are not order dependent."""
        bounds_rect = self.draw_aalines(closed=1, points=((0, 0), (1, 1), (2, 2)), color=(10, 20, 30), surface=pygame.Surface((3, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__args_missing(self):
        """Ensures draw aalines detects any missing required args."""
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color, 0)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines()

    def test_aalines__kwargs_missing(self):
        """Ensures draw aalines detects any missing required kwargs."""
        kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'closed': 1, 'points': ((2, 2), (1, 1))}
        for name in ('points', 'closed', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**invalid_kwargs)

    def test_aalines__arg_invalid_types(self):
        """Ensures draw aalines detects invalid arg types."""
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        closed = 0
        points = ((1, 2), (2, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color, closed, points, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color, closed, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color, InvalidBool(), points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, 2.3, closed, points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines((1, 2, 3, 4), color, closed, points)

    def test_aalines__kwarg_invalid_types(self):
        """Ensures draw aalines detects invalid kwarg types."""
        valid_kwargs = {'surface': pygame.Surface((3, 3)), 'color': pygame.Color('green'), 'closed': False, 'points': ((1, 2), (2, 1))}
        invalid_kwargs = {'surface': pygame.Surface, 'color': 2.3, 'closed': InvalidBool(), 'points': (0, 0, 0)}
        for kwarg in ('surface', 'color', 'closed', 'points'):
            kwargs = dict(valid_kwargs)
            kwargs[kwarg] = invalid_kwargs[kwarg]
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__kwarg_invalid_name(self):
        """Ensures draw aalines detects invalid kwarg names."""
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        closed = 1
        points = ((1, 2), (2, 1))
        kwargs_list = [{'surface': surface, 'color': color, 'closed': closed, 'points': points, 'invalid': 1}, {'surface': surface, 'color': color, 'closed': closed, 'points': points, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__args_and_kwargs(self):
        """Ensures draw aalines accepts a combination of args/kwargs"""
        surface = pygame.Surface((3, 2))
        color = (255, 255, 0, 0)
        closed = 0
        points = ((1, 2), (2, 1))
        kwargs = {'surface': surface, 'color': color, 'closed': closed, 'points': points}
        for name in ('surface', 'color', 'closed', 'points'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_aalines(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_aalines(surface, color, **kwargs)
            elif 'closed' == name:
                bounds_rect = self.draw_aalines(surface, color, closed, **kwargs)
            elif 'points' == name:
                bounds_rect = self.draw_aalines(surface, color, closed, points, **kwargs)
            else:
                bounds_rect = self.draw_aalines(surface, color, closed, points, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__valid_points_format(self):
        """Ensures draw aalines accepts different points formats."""
        expected_color = (10, 20, 30, 255)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'closed': False, 'points': None}
        point_types = ((tuple, tuple, tuple, tuple), (list, list, list, list), (Vector2, Vector2, Vector2, Vector2), (list, Vector2, tuple, Vector2))
        point_values = (((1, 1), (2, 1), (2, 2), (1, 2)), ((1, 1), (2.2, 1), (2.1, 2.2), (1, 2.1)))
        seq_types = (tuple, list)
        for point_type in point_types:
            for values in point_values:
                check_pos = values[0]
                points = [point_type[i](pt) for i, pt in enumerate(values)]
                for seq_type in seq_types:
                    surface.fill(surface_color)
                    kwargs['points'] = seq_type(points)
                    bounds_rect = self.draw_aalines(**kwargs)
                    self.assertEqual(surface.get_at(check_pos), expected_color)
                    self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__invalid_points_formats(self):
        """Ensures draw aalines handles invalid points formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None}
        points_fmts = (((1, 1), (2,)), ((1, 1), (2, 2, 2)), ((1, 1), (2, '2')), ((1, 1), {2, 3}), ((1, 1), dict(((2, 2), (3, 3)))), {(1, 1), (1, 2)}, dict(((1, 1), (4, 4))))
        for points in points_fmts:
            kwargs['points'] = points
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__invalid_points_values(self):
        """Ensures draw aalines handles invalid points values correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None}
        for points in ([], ((1, 1),)):
            for seq_type in (tuple, list):
                kwargs['points'] = seq_type(points)
                with self.assertRaises(ValueError):
                    bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__valid_closed_values(self):
        """Ensures draw aalines accepts different closed values."""
        line_color = pygame.Color('blue')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((5, 5))
        pos = (1, 3)
        kwargs = {'surface': surface, 'color': line_color, 'closed': None, 'points': ((1, 1), (4, 1), (4, 4), (1, 4))}
        true_values = (-7, 1, 10, '2', 3.1, (4,), [5], True)
        false_values = (None, '', 0, (), [], False)
        for closed in true_values + false_values:
            surface.fill(surface_color)
            kwargs['closed'] = closed
            expected_color = line_color if closed else surface_color
            bounds_rect = self.draw_aalines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__valid_color_formats(self):
        """Ensures draw aalines accepts different color formats."""
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': None, 'closed': False, 'points': (pos, (2, 1))}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_aalines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__invalid_color_formats(self):
        """Ensures draw aalines handles invalid color formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'closed': False, 'points': ((1, 1), (1, 2))}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__color(self):
        """Tests if the aalines drawn are the correct color.

        Draws aalines around the border of the given surface and checks if all
        borders of the surface only contain the given color.
        """
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_aalines(surface, expected_color, True, corners(surface))
                for pos, color in border_pos_and_color(surface):
                    self.assertEqual(color, expected_color, f'pos={pos}')

    def test_aalines__gaps(self):
        """Tests if the aalines drawn contain any gaps.

        Draws aalines around the border of the given surface and checks if
        all borders of the surface contain any gaps.

        See: #512
        """
        expected_color = (255, 255, 255)
        for surface in self._create_surfaces():
            self.draw_aalines(surface, expected_color, True, corners(surface))
            for pos, color in border_pos_and_color(surface):
                self.assertEqual(color, expected_color, f'pos={pos}')

    def test_aalines__bounding_rect(self):
        """Ensures draw aalines returns the correct bounding rect.

        Tests lines with endpoints on and off the surface and blending
        enabled and disabled.
        """
        line_color = pygame.Color('red')
        surf_color = pygame.Color('blue')
        width = height = 30
        pos_rect = pygame.Rect((0, 0), (width, height))
        for size in ((width + 5, height + 5), (width - 5, height - 5)):
            surface = pygame.Surface(size, 0, 32)
            surf_rect = surface.get_rect()
            for pos in rect_corners_mids_and_center(surf_rect):
                pos_rect.center = pos
                pts = (pos_rect.midleft, pos_rect.midtop, pos_rect.midright)
                pos = pts[0]
                for closed in (True, False):
                    surface.fill(surf_color)
                    bounding_rect = self.draw_aalines(surface, line_color, closed, pts)
                    expected_rect = create_bounding_rect(surface, surf_color, pos)
                    self.assertEqual(bounding_rect, expected_rect)

    def test_aalines__surface_clip(self):
        """Ensures draw aalines respects a surface's clip area."""
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
            pts = (pos_rect.midtop, pos_rect.center, pos_rect.midbottom)
            for closed in (True, False):
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_aalines(surface, aaline_color, closed, pts)
                expected_pts = get_color_points(surface, surface_color, clip_rect, False)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_aalines(surface, aaline_color, closed, pts)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        self.assertNotEqual(surface.get_at(pt), surface_color, pt)
                    else:
                        self.assertEqual(surface.get_at(pt), surface_color, pt)
                surface.unlock()