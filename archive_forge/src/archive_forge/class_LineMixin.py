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
class LineMixin(BaseLineMixin):
    """Mixin test for drawing a single line.

    This class contains all the general single line drawing tests.
    """

    def test_line__args(self):
        """Ensures draw line accepts the correct args."""
        bounds_rect = self.draw_line(pygame.Surface((3, 3)), (0, 10, 0, 50), (0, 0), (1, 1), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__args_without_width(self):
        """Ensures draw line accepts the args without a width."""
        bounds_rect = self.draw_line(pygame.Surface((2, 2)), (0, 0, 0, 50), (0, 0), (2, 2))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__kwargs(self):
        """Ensures draw line accepts the correct kwargs
        with and without a width arg.
        """
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        start_pos = (1, 1)
        end_pos = (2, 2)
        kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': 1}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_line(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__kwargs_order_independent(self):
        """Ensures draw line's kwargs are not order dependent."""
        bounds_rect = self.draw_line(start_pos=(1, 2), end_pos=(2, 1), width=2, color=(10, 20, 30), surface=pygame.Surface((3, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__args_missing(self):
        """Ensures draw line detects any missing required args."""
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color, (0, 0))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line()

    def test_line__kwargs_missing(self):
        """Ensures draw line detects any missing required kwargs."""
        kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'start_pos': (2, 1), 'end_pos': (2, 2), 'width': 1}
        for name in ('end_pos', 'start_pos', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**invalid_kwargs)

    def test_line__arg_invalid_types(self):
        """Ensures draw line detects invalid arg types."""
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        start_pos = (0, 1)
        end_pos = (1, 2)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color, start_pos, end_pos, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color, start_pos, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color, (1,), end_pos)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, 2.3, start_pos, end_pos)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line((1, 2, 3, 4), color, start_pos, end_pos)

    def test_line__kwarg_invalid_types(self):
        """Ensures draw line detects invalid kwarg types."""
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        start_pos = (1, 0)
        end_pos = (2, 0)
        width = 1
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': width}, {'surface': surface, 'color': 2.3, 'start_pos': start_pos, 'end_pos': end_pos, 'width': width}, {'surface': surface, 'color': color, 'start_pos': (0, 0, 0), 'end_pos': end_pos, 'width': width}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': (0,), 'width': width}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': 1.2}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__kwarg_invalid_name(self):
        """Ensures draw line detects invalid kwarg names."""
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        start_pos = (1, 1)
        end_pos = (2, 0)
        kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__args_and_kwargs(self):
        """Ensures draw line accepts a combination of args/kwargs"""
        surface = pygame.Surface((3, 2))
        color = (255, 255, 0, 0)
        start_pos = (0, 1)
        end_pos = (1, 2)
        width = 0
        kwargs = {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': width}
        for name in ('surface', 'color', 'start_pos', 'end_pos', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_line(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_line(surface, color, **kwargs)
            elif 'start_pos' == name:
                bounds_rect = self.draw_line(surface, color, start_pos, **kwargs)
            elif 'end_pos' == name:
                bounds_rect = self.draw_line(surface, color, start_pos, end_pos, **kwargs)
            else:
                bounds_rect = self.draw_line(surface, color, start_pos, end_pos, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__valid_width_values(self):
        """Ensures draw line accepts different width values."""
        line_color = pygame.Color('yellow')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        pos = (2, 1)
        kwargs = {'surface': surface, 'color': line_color, 'start_pos': pos, 'end_pos': (2, 2), 'width': None}
        for width in (-100, -10, -1, 0, 1, 10, 100):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = line_color if width > 0 else surface_color
            bounds_rect = self.draw_line(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__valid_start_pos_formats(self):
        """Ensures draw line accepts different start_pos formats."""
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'start_pos': None, 'end_pos': (2, 2), 'width': 2}
        x, y = (2, 1)
        for start_pos in ((x, y), (x + 0.1, y), (x, y + 0.1), (x + 0.1, y + 0.1)):
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['start_pos'] = seq_type(start_pos)
                bounds_rect = self.draw_line(**kwargs)
                self.assertEqual(surface.get_at((x, y)), expected_color)
                self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__valid_end_pos_formats(self):
        """Ensures draw line accepts different end_pos formats."""
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'start_pos': (2, 1), 'end_pos': None, 'width': 2}
        x, y = (2, 2)
        for end_pos in ((x, y), (x + 0.2, y), (x, y + 0.2), (x + 0.2, y + 0.2)):
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['end_pos'] = seq_type(end_pos)
                bounds_rect = self.draw_line(**kwargs)
                self.assertEqual(surface.get_at((x, y)), expected_color)
                self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__invalid_start_pos_formats(self):
        """Ensures draw line handles invalid start_pos formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': None, 'end_pos': (2, 2), 'width': 1}
        start_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
        for start_pos in start_pos_fmts:
            kwargs['start_pos'] = start_pos
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__invalid_end_pos_formats(self):
        """Ensures draw line handles invalid end_pos formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': (2, 2), 'end_pos': None, 'width': 1}
        end_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
        for end_pos in end_pos_fmts:
            kwargs['end_pos'] = end_pos
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__valid_color_formats(self):
        """Ensures draw line accepts different color formats."""
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': None, 'start_pos': pos, 'end_pos': (2, 1), 'width': 3}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_line(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__invalid_color_formats(self):
        """Ensures draw line handles invalid color formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'start_pos': (1, 1), 'end_pos': (2, 1), 'width': 1}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__color(self):
        """Tests if the line drawn is the correct color."""
        pos = (0, 0)
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_line(surface, expected_color, pos, (1, 0))
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_line__color_with_thickness(self):
        """Ensures a thick line is drawn using the correct color."""
        from_x = 5
        to_x = 10
        y = 5
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_line(surface, expected_color, (from_x, y), (to_x, y), 5)
                for pos in ((x, y + i) for i in (-2, 0, 2) for x in (from_x, to_x)):
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_line__gaps(self):
        """Tests if the line drawn contains any gaps."""
        expected_color = (255, 255, 255)
        for surface in self._create_surfaces():
            width = surface.get_width()
            self.draw_line(surface, expected_color, (0, 0), (width - 1, 0))
            for x in range(width):
                pos = (x, 0)
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_line__gaps_with_thickness(self):
        """Ensures a thick line is drawn without any gaps."""
        expected_color = (255, 255, 255)
        thickness = 5
        for surface in self._create_surfaces():
            width = surface.get_width() - 1
            h = width // 5
            w = h * 5
            self.draw_line(surface, expected_color, (0, 5), (w, 5 + h), thickness)
            for x in range(w + 1):
                for y in range(3, 8):
                    pos = (x, y + (x + 2) // 5)
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_line__bounding_rect(self):
        """Ensures draw line returns the correct bounding rect.

        Tests lines with endpoints on and off the surface and a range of
        width/thickness values.
        """
        if isinstance(self, PythonDrawTestCase):
            self.skipTest('bounding rects not supported in draw_py.draw_line')
        line_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        width = height = 30
        helper_rect = pygame.Rect((0, 0), (width, height))
        for size in ((width + 5, height + 5), (width - 5, height - 5)):
            surface = pygame.Surface(size, 0, 32)
            surf_rect = surface.get_rect()
            for pos in rect_corners_mids_and_center(surf_rect):
                helper_rect.center = pos
                for thickness in range(-1, 5):
                    for start, end in self._rect_lines(helper_rect):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_line(surface, line_color, start, end, thickness)
                        if 0 < thickness:
                            expected_rect = create_bounding_rect(surface, surf_color, start)
                        else:
                            expected_rect = pygame.Rect(start, (0, 0))
                        self.assertEqual(bounding_rect, expected_rect, 'start={}, end={}, size={}, thickness={}'.format(start, end, size, thickness))

    def test_line__surface_clip(self):
        """Ensures draw line respects a surface's clip area."""
        surfw = surfh = 30
        line_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for thickness in (1, 3):
            for center in rect_corners_mids_and_center(clip_rect):
                pos_rect.center = center
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_line(surface, line_color, pos_rect.midtop, pos_rect.midbottom, thickness)
                expected_pts = get_color_points(surface, line_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_line(surface, line_color, pos_rect.midtop, pos_rect.midbottom, thickness)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = line_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()