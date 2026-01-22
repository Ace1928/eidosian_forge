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
class LinesMixin(BaseLineMixin):
    """Mixin test for drawing lines.

    This class contains all the general lines drawing tests.
    """

    def test_lines__args(self):
        """Ensures draw lines accepts the correct args."""
        bounds_rect = self.draw_lines(pygame.Surface((3, 3)), (0, 10, 0, 50), False, ((0, 0), (1, 1)), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__args_without_width(self):
        """Ensures draw lines accepts the args without a width."""
        bounds_rect = self.draw_lines(pygame.Surface((2, 2)), (0, 0, 0, 50), False, ((0, 0), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__kwargs(self):
        """Ensures draw lines accepts the correct kwargs
        with and without a width arg.
        """
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        points = ((0, 0), (1, 1), (2, 2))
        kwargs_list = [{'surface': surface, 'color': color, 'closed': False, 'points': points, 'width': 1}, {'surface': surface, 'color': color, 'closed': False, 'points': points}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_lines(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__kwargs_order_independent(self):
        """Ensures draw lines's kwargs are not order dependent."""
        bounds_rect = self.draw_lines(closed=1, points=((0, 0), (1, 1), (2, 2)), width=2, color=(10, 20, 30), surface=pygame.Surface((3, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__args_missing(self):
        """Ensures draw lines detects any missing required args."""
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color, 0)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines()

    def test_lines__kwargs_missing(self):
        """Ensures draw lines detects any missing required kwargs."""
        kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'closed': 1, 'points': ((2, 2), (1, 1)), 'width': 1}
        for name in ('points', 'closed', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**invalid_kwargs)

    def test_lines__arg_invalid_types(self):
        """Ensures draw lines detects invalid arg types."""
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        closed = 0
        points = ((1, 2), (2, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color, closed, points, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color, closed, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color, InvalidBool(), points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, 2.3, closed, points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines((1, 2, 3, 4), color, closed, points)

    def test_lines__kwarg_invalid_types(self):
        """Ensures draw lines detects invalid kwarg types."""
        valid_kwargs = {'surface': pygame.Surface((3, 3)), 'color': pygame.Color('green'), 'closed': False, 'points': ((1, 2), (2, 1)), 'width': 1}
        invalid_kwargs = {'surface': pygame.Surface, 'color': 2.3, 'closed': InvalidBool(), 'points': (0, 0, 0), 'width': 1.2}
        for kwarg in ('surface', 'color', 'closed', 'points', 'width'):
            kwargs = dict(valid_kwargs)
            kwargs[kwarg] = invalid_kwargs[kwarg]
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**kwargs)

    def test_lines__kwarg_invalid_name(self):
        """Ensures draw lines detects invalid kwarg names."""
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        closed = 1
        points = ((1, 2), (2, 1))
        kwargs_list = [{'surface': surface, 'color': color, 'closed': closed, 'points': points, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'closed': closed, 'points': points, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**kwargs)

    def test_lines__args_and_kwargs(self):
        """Ensures draw lines accepts a combination of args/kwargs"""
        surface = pygame.Surface((3, 2))
        color = (255, 255, 0, 0)
        closed = 0
        points = ((1, 2), (2, 1))
        width = 1
        kwargs = {'surface': surface, 'color': color, 'closed': closed, 'points': points, 'width': width}
        for name in ('surface', 'color', 'closed', 'points', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_lines(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_lines(surface, color, **kwargs)
            elif 'closed' == name:
                bounds_rect = self.draw_lines(surface, color, closed, **kwargs)
            elif 'points' == name:
                bounds_rect = self.draw_lines(surface, color, closed, points, **kwargs)
            else:
                bounds_rect = self.draw_lines(surface, color, closed, points, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__valid_width_values(self):
        """Ensures draw lines accepts different width values."""
        line_color = pygame.Color('yellow')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': line_color, 'closed': False, 'points': (pos, (2, 1)), 'width': None}
        for width in (-100, -10, -1, 0, 1, 10, 100):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = line_color if width > 0 else surface_color
            bounds_rect = self.draw_lines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__valid_points_format(self):
        """Ensures draw lines accepts different points formats."""
        expected_color = (10, 20, 30, 255)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'closed': False, 'points': None, 'width': 1}
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
                    bounds_rect = self.draw_lines(**kwargs)
                    self.assertEqual(surface.get_at(check_pos), expected_color)
                    self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__invalid_points_formats(self):
        """Ensures draw lines handles invalid points formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None, 'width': 1}
        points_fmts = (((1, 1), (2,)), ((1, 1), (2, 2, 2)), ((1, 1), (2, '2')), ((1, 1), {2, 3}), ((1, 1), dict(((2, 2), (3, 3)))), {(1, 1), (1, 2)}, dict(((1, 1), (4, 4))))
        for points in points_fmts:
            kwargs['points'] = points
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**kwargs)

    def test_lines__invalid_points_values(self):
        """Ensures draw lines handles invalid points values correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None, 'width': 1}
        for points in ([], ((1, 1),)):
            for seq_type in (tuple, list):
                kwargs['points'] = seq_type(points)
                with self.assertRaises(ValueError):
                    bounds_rect = self.draw_lines(**kwargs)

    def test_lines__valid_closed_values(self):
        """Ensures draw lines accepts different closed values."""
        line_color = pygame.Color('blue')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        pos = (1, 2)
        kwargs = {'surface': surface, 'color': line_color, 'closed': None, 'points': ((1, 1), (3, 1), (3, 3), (1, 3)), 'width': 1}
        true_values = (-7, 1, 10, '2', 3.1, (4,), [5], True)
        false_values = (None, '', 0, (), [], False)
        for closed in true_values + false_values:
            surface.fill(surface_color)
            kwargs['closed'] = closed
            expected_color = line_color if closed else surface_color
            bounds_rect = self.draw_lines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__valid_color_formats(self):
        """Ensures draw lines accepts different color formats."""
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': None, 'closed': False, 'points': (pos, (2, 1)), 'width': 3}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_lines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__invalid_color_formats(self):
        """Ensures draw lines handles invalid color formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'closed': False, 'points': ((1, 1), (1, 2)), 'width': 1}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**kwargs)

    def test_lines__color(self):
        """Tests if the lines drawn are the correct color.

        Draws lines around the border of the given surface and checks if all
        borders of the surface only contain the given color.
        """
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_lines(surface, expected_color, True, corners(surface))
                for pos, color in border_pos_and_color(surface):
                    self.assertEqual(color, expected_color, f'pos={pos}')

    def test_lines__color_with_thickness(self):
        """Ensures thick lines are drawn using the correct color."""
        x_left = y_top = 5
        for surface in self._create_surfaces():
            x_right = surface.get_width() - 5
            y_bottom = surface.get_height() - 5
            endpoints = ((x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom))
            for expected_color in self.COLORS:
                self.draw_lines(surface, expected_color, True, endpoints, 3)
                for t in (-1, 0, 1):
                    for x in range(x_left, x_right + 1):
                        for y in (y_top, y_bottom):
                            pos = (x, y + t)
                            self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')
                    for y in range(y_top, y_bottom + 1):
                        for x in (x_left, x_right):
                            pos = (x + t, y)
                            self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_lines__gaps(self):
        """Tests if the lines drawn contain any gaps.

        Draws lines around the border of the given surface and checks if
        all borders of the surface contain any gaps.
        """
        expected_color = (255, 255, 255)
        for surface in self._create_surfaces():
            self.draw_lines(surface, expected_color, True, corners(surface))
            for pos, color in border_pos_and_color(surface):
                self.assertEqual(color, expected_color, f'pos={pos}')

    def test_lines__gaps_with_thickness(self):
        """Ensures thick lines are drawn without any gaps."""
        expected_color = (255, 255, 255)
        x_left = y_top = 5
        for surface in self._create_surfaces():
            h = (surface.get_width() - 11) // 5
            w = h * 5
            x_right = x_left + w
            y_bottom = y_top + h
            endpoints = ((x_left, y_top), (x_right, y_top), (x_right, y_bottom))
            self.draw_lines(surface, expected_color, True, endpoints, 3)
            for x in range(x_left, x_right + 1):
                for t in (-1, 0, 1):
                    pos = (x, y_top + t)
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')
                    pos = (x, y_top + t + (x - 3) // 5)
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')
            for y in range(y_top, y_bottom + 1):
                for t in (-1, 0, 1):
                    pos = (x_right + t, y)
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_lines__bounding_rect(self):
        """Ensures draw lines returns the correct bounding rect.

        Tests lines with endpoints on and off the surface and a range of
        width/thickness values.
        """
        line_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        width = height = 30
        pos_rect = pygame.Rect((0, 0), (width, height))
        for size in ((width + 5, height + 5), (width - 5, height - 5)):
            surface = pygame.Surface(size, 0, 32)
            surf_rect = surface.get_rect()
            for pos in rect_corners_mids_and_center(surf_rect):
                pos_rect.center = pos
                pts = (pos_rect.midleft, pos_rect.midtop, pos_rect.midright)
                pos = pts[0]
                for thickness in range(-1, 5):
                    for closed in (True, False):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_lines(surface, line_color, closed, pts, thickness)
                        if 0 < thickness:
                            expected_rect = create_bounding_rect(surface, surf_color, pos)
                        else:
                            expected_rect = pygame.Rect(pos, (0, 0))
                        self.assertEqual(bounding_rect, expected_rect)

    def test_lines__surface_clip(self):
        """Ensures draw lines respects a surface's clip area."""
        surfw = surfh = 30
        line_color = pygame.Color('red')
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
                for thickness in (1, 3):
                    surface.set_clip(None)
                    surface.fill(surface_color)
                    self.draw_lines(surface, line_color, closed, pts, thickness)
                    expected_pts = get_color_points(surface, line_color, clip_rect)
                    surface.fill(surface_color)
                    surface.set_clip(clip_rect)
                    self.draw_lines(surface, line_color, closed, pts, thickness)
                    surface.lock()
                    for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                        if pt in expected_pts:
                            expected_color = line_color
                        else:
                            expected_color = surface_color
                        self.assertEqual(surface.get_at(pt), expected_color, pt)
                    surface.unlock()