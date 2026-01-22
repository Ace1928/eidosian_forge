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
class DrawEllipseMixin:
    """Mixin tests for drawing ellipses.

    This class contains all the general ellipse drawing tests.
    """

    def test_ellipse__args(self):
        """Ensures draw ellipse accepts the correct args."""
        bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), pygame.Rect((0, 0), (3, 2)), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__args_without_width(self):
        """Ensures draw ellipse accepts the args without a width."""
        bounds_rect = self.draw_ellipse(pygame.Surface((2, 2)), (1, 1, 1, 99), pygame.Rect((1, 1), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__args_with_negative_width(self):
        """Ensures draw ellipse accepts the args with negative width."""
        bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), pygame.Rect((2, 3), (3, 2)), -1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        self.assertEqual(bounds_rect, pygame.Rect(2, 3, 0, 0))

    def test_ellipse__args_with_width_gt_radius(self):
        """Ensures draw ellipse accepts the args with
        width > rect.w // 2 and width > rect.h // 2.
        """
        rect = pygame.Rect((0, 0), (4, 4))
        bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), rect, rect.w // 2 + 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), rect, rect.h // 2 + 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__kwargs(self):
        """Ensures draw ellipse accepts the correct kwargs
        with and without a width arg.
        """
        kwargs_list = [{'surface': pygame.Surface((4, 4)), 'color': pygame.Color('yellow'), 'rect': pygame.Rect((0, 0), (3, 2)), 'width': 1}, {'surface': pygame.Surface((2, 1)), 'color': (0, 10, 20), 'rect': (0, 0, 1, 1)}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_ellipse(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__kwargs_order_independent(self):
        """Ensures draw ellipse's kwargs are not order dependent."""
        bounds_rect = self.draw_ellipse(color=(1, 2, 3), surface=pygame.Surface((3, 2)), width=0, rect=pygame.Rect((1, 0), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__args_missing(self):
        """Ensures draw ellipse detects any missing required args."""
        surface = pygame.Surface((1, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface, pygame.Color('red'))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse()

    def test_ellipse__kwargs_missing(self):
        """Ensures draw ellipse detects any missing required kwargs."""
        kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'rect': pygame.Rect((1, 0), (2, 2)), 'width': 2}
        for name in ('rect', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_ellipse(**invalid_kwargs)

    def test_ellipse__arg_invalid_types(self):
        """Ensures draw ellipse detects invalid arg types."""
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        rect = pygame.Rect((1, 1), (1, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface, color, rect, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface, color, (1, 2, 3, 4, 5), 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface, 2.3, rect, 0)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(rect, color, rect, 2)

    def test_ellipse__kwarg_invalid_types(self):
        """Ensures draw ellipse detects invalid kwarg types."""
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        rect = pygame.Rect((0, 1), (1, 1))
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'rect': rect, 'width': 1}, {'surface': surface, 'color': 2.3, 'rect': rect, 'width': 1}, {'surface': surface, 'color': color, 'rect': (0, 0, 0), 'width': 1}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1.1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_ellipse(**kwargs)

    def test_ellipse__kwarg_invalid_name(self):
        """Ensures draw ellipse detects invalid kwarg names."""
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        rect = pygame.Rect((0, 1), (2, 2))
        kwargs_list = [{'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'rect': rect, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_ellipse(**kwargs)

    def test_ellipse__args_and_kwargs(self):
        """Ensures draw ellipse accepts a combination of args/kwargs"""
        surface = pygame.Surface((3, 1))
        color = (255, 255, 0, 0)
        rect = pygame.Rect((1, 0), (2, 1))
        width = 0
        kwargs = {'surface': surface, 'color': color, 'rect': rect, 'width': width}
        for name in ('surface', 'color', 'rect', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_ellipse(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_ellipse(surface, color, **kwargs)
            elif 'rect' == name:
                bounds_rect = self.draw_ellipse(surface, color, rect, **kwargs)
            else:
                bounds_rect = self.draw_ellipse(surface, color, rect, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__valid_width_values(self):
        """Ensures draw ellipse accepts different width values."""
        pos = (1, 1)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        color = (10, 20, 30, 255)
        kwargs = {'surface': surface, 'color': color, 'rect': pygame.Rect(pos, (3, 2)), 'width': None}
        for width in (-1000, -10, -1, 0, 1, 10, 1000):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = color if width >= 0 else surface_color
            bounds_rect = self.draw_ellipse(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__valid_rect_formats(self):
        """Ensures draw ellipse accepts different rect formats."""
        pos = (1, 1)
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'rect': None, 'width': 0}
        rects = (pygame.Rect(pos, (1, 3)), (pos, (2, 1)), (pos[0], pos[1], 1, 1))
        for rect in rects:
            surface.fill(surface_color)
            kwargs['rect'] = rect
            bounds_rect = self.draw_ellipse(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__valid_color_formats(self):
        """Ensures draw ellipse accepts different color formats."""
        pos = (1, 1)
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (1, 2)), 'width': 0}
        reds = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in reds:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_ellipse(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__invalid_color_formats(self):
        """Ensures draw ellipse handles invalid color formats correctly."""
        pos = (1, 1)
        surface = pygame.Surface((4, 3))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (2, 2)), 'width': 1}
        for expected_color in (2.3, surface):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_ellipse(**kwargs)

    def test_ellipse(self):
        """Tests ellipses of differing sizes on surfaces of differing sizes.

        Checks if the number of sides touching the border of the surface is
        correct.
        """
        left_top = [(0, 0), (1, 0), (0, 1), (1, 1)]
        sizes = [(4, 4), (5, 4), (4, 5), (5, 5)]
        color = (1, 13, 24, 255)

        def same_size(width, height, border_width):
            """Test for ellipses with the same size as the surface."""
            surface = pygame.Surface((width, height))
            self.draw_ellipse(surface, color, (0, 0, width, height), border_width)
            borders = get_border_values(surface, width, height)
            for border in borders:
                self.assertTrue(color in border)

        def not_same_size(width, height, border_width, left, top):
            """Test for ellipses that aren't the same size as the surface."""
            surface = pygame.Surface((width, height))
            self.draw_ellipse(surface, color, (left, top, width - 1, height - 1), border_width)
            borders = get_border_values(surface, width, height)
            sides_touching = [color in border for border in borders].count(True)
            self.assertEqual(sides_touching, 2)
        for width, height in sizes:
            for border_width in (0, 1):
                same_size(width, height, border_width)
                for left, top in left_top:
                    not_same_size(width, height, border_width, left, top)

    def test_ellipse__big_ellipse(self):
        """Test for big ellipse that could overflow in algorithm"""
        width = 1025
        height = 1025
        border = 1
        x_value_test = int(0.4 * height)
        y_value_test = int(0.4 * height)
        surface = pygame.Surface((width, height))
        self.draw_ellipse(surface, (255, 0, 0), (0, 0, width, height), border)
        colored_pixels = 0
        for y in range(height):
            if surface.get_at((x_value_test, y)) == (255, 0, 0):
                colored_pixels += 1
        for x in range(width):
            if surface.get_at((x, y_value_test)) == (255, 0, 0):
                colored_pixels += 1
        self.assertEqual(colored_pixels, border * 4)

    def test_ellipse__thick_line(self):
        """Ensures a thick lined ellipse is drawn correctly."""
        ellipse_color = pygame.Color('yellow')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((40, 40))
        rect = pygame.Rect((0, 0), (31, 23))
        rect.center = surface.get_rect().center
        for thickness in range(1, min(*rect.size) // 2 - 2):
            surface.fill(surface_color)
            self.draw_ellipse(surface, ellipse_color, rect, thickness)
            surface.lock()
            x = rect.centerx
            y_start = rect.top
            y_end = rect.top + thickness - 1
            for y in range(y_start, y_end + 1):
                self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
            self.assertEqual(surface.get_at((x, y_start - 1)), surface_color, thickness)
            self.assertEqual(surface.get_at((x, y_end + 1)), surface_color, thickness)
            x = rect.centerx
            y_start = rect.bottom - thickness
            y_end = rect.bottom - 1
            for y in range(y_start, y_end + 1):
                self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
            self.assertEqual(surface.get_at((x, y_start - 1)), surface_color, thickness)
            self.assertEqual(surface.get_at((x, y_end + 1)), surface_color, thickness)
            x_start = rect.left
            x_end = rect.left + thickness - 1
            y = rect.centery
            for x in range(x_start, x_end + 1):
                self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
            self.assertEqual(surface.get_at((x_start - 1, y)), surface_color, thickness)
            self.assertEqual(surface.get_at((x_end + 1, y)), surface_color, thickness)
            x_start = rect.right - thickness
            x_end = rect.right - 1
            y = rect.centery
            for x in range(x_start, x_end + 1):
                self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
            self.assertEqual(surface.get_at((x_start - 1, y)), surface_color, thickness)
            self.assertEqual(surface.get_at((x_end + 1, y)), surface_color, thickness)
            surface.unlock()

    def test_ellipse__no_holes(self):
        width = 80
        height = 70
        surface = pygame.Surface((width + 1, height))
        rect = pygame.Rect(0, 0, width, height)
        for thickness in range(1, 37, 5):
            surface.fill('BLACK')
            self.draw_ellipse(surface, 'RED', rect, thickness)
            for y in range(height):
                number_of_changes = 0
                drawn_pixel = False
                for x in range(width + 1):
                    if not drawn_pixel and surface.get_at((x, y)) == pygame.Color('RED') or (drawn_pixel and surface.get_at((x, y)) == pygame.Color('BLACK')):
                        drawn_pixel = not drawn_pixel
                        number_of_changes += 1
                if y < thickness or y > height - thickness - 1:
                    self.assertEqual(number_of_changes, 2)
                else:
                    self.assertEqual(number_of_changes, 4)

    def test_ellipse__max_width(self):
        """Ensures an ellipse with max width (and greater) is drawn correctly."""
        ellipse_color = pygame.Color('yellow')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((40, 40))
        rect = pygame.Rect((0, 0), (31, 21))
        rect.center = surface.get_rect().center
        max_thickness = (min(*rect.size) + 1) // 2
        for thickness in range(max_thickness, max_thickness + 3):
            surface.fill(surface_color)
            self.draw_ellipse(surface, ellipse_color, rect, thickness)
            surface.lock()
            for y in range(rect.top, rect.bottom):
                self.assertEqual(surface.get_at((rect.centerx, y)), ellipse_color)
            for x in range(rect.left, rect.right):
                self.assertEqual(surface.get_at((x, rect.centery)), ellipse_color)
            self.assertEqual(surface.get_at((rect.centerx, rect.top - 1)), surface_color)
            self.assertEqual(surface.get_at((rect.centerx, rect.bottom + 1)), surface_color)
            self.assertEqual(surface.get_at((rect.left - 1, rect.centery)), surface_color)
            self.assertEqual(surface.get_at((rect.right + 1, rect.centery)), surface_color)
            surface.unlock()

    def _check_1_pixel_sized_ellipse(self, surface, collide_rect, surface_color, ellipse_color):
        surf_w, surf_h = surface.get_size()
        surface.lock()
        for pos in ((x, y) for y in range(surf_h) for x in range(surf_w)):
            if collide_rect.collidepoint(pos):
                expected_color = ellipse_color
            else:
                expected_color = surface_color
            self.assertEqual(surface.get_at(pos), expected_color, f'collide_rect={collide_rect}, pos={pos}')
        surface.unlock()

    def test_ellipse__1_pixel_width(self):
        """Ensures an ellipse with a width of 1 is drawn correctly.

        An ellipse with a width of 1 pixel is a vertical line.
        """
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surf_w, surf_h = (10, 20)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (1, 0))
        collide_rect = rect.copy()
        off_left = -1
        off_right = surf_w
        off_bottom = surf_h
        center_x = surf_w // 2
        center_y = surf_h // 2
        for ellipse_h in range(6, 10):
            collide_rect.h = ellipse_h
            rect.h = ellipse_h
            off_top = -(ellipse_h + 1)
            half_off_top = -(ellipse_h // 2)
            half_off_bottom = surf_h - ellipse_h // 2
            positions = ((off_left, off_top), (off_left, half_off_top), (off_left, center_y), (off_left, half_off_bottom), (off_left, off_bottom), (center_x, off_top), (center_x, half_off_top), (center_x, center_y), (center_x, half_off_bottom), (center_x, off_bottom), (off_right, off_top), (off_right, half_off_top), (off_right, center_y), (off_right, half_off_bottom), (off_right, off_bottom))
            for rect_pos in positions:
                surface.fill(surface_color)
                rect.topleft = rect_pos
                collide_rect.topleft = rect_pos
                self.draw_ellipse(surface, ellipse_color, rect)
                self._check_1_pixel_sized_ellipse(surface, collide_rect, surface_color, ellipse_color)

    def test_ellipse__1_pixel_width_spanning_surface(self):
        """Ensures an ellipse with a width of 1 is drawn correctly
        when spanning the height of the surface.

        An ellipse with a width of 1 pixel is a vertical line.
        """
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surf_w, surf_h = (10, 20)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (1, surf_h + 2))
        positions = ((-1, -1), (0, -1), (surf_w // 2, -1), (surf_w - 1, -1), (surf_w, -1))
        for rect_pos in positions:
            surface.fill(surface_color)
            rect.topleft = rect_pos
            self.draw_ellipse(surface, ellipse_color, rect)
            self._check_1_pixel_sized_ellipse(surface, rect, surface_color, ellipse_color)

    def test_ellipse__1_pixel_height(self):
        """Ensures an ellipse with a height of 1 is drawn correctly.

        An ellipse with a height of 1 pixel is a horizontal line.
        """
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surf_w, surf_h = (20, 10)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (0, 1))
        collide_rect = rect.copy()
        off_right = surf_w
        off_top = -1
        off_bottom = surf_h
        center_x = surf_w // 2
        center_y = surf_h // 2
        for ellipse_w in range(6, 10):
            collide_rect.w = ellipse_w
            rect.w = ellipse_w
            off_left = -(ellipse_w + 1)
            half_off_left = -(ellipse_w // 2)
            half_off_right = surf_w - ellipse_w // 2
            positions = ((off_left, off_top), (half_off_left, off_top), (center_x, off_top), (half_off_right, off_top), (off_right, off_top), (off_left, center_y), (half_off_left, center_y), (center_x, center_y), (half_off_right, center_y), (off_right, center_y), (off_left, off_bottom), (half_off_left, off_bottom), (center_x, off_bottom), (half_off_right, off_bottom), (off_right, off_bottom))
            for rect_pos in positions:
                surface.fill(surface_color)
                rect.topleft = rect_pos
                collide_rect.topleft = rect_pos
                self.draw_ellipse(surface, ellipse_color, rect)
                self._check_1_pixel_sized_ellipse(surface, collide_rect, surface_color, ellipse_color)

    def test_ellipse__1_pixel_height_spanning_surface(self):
        """Ensures an ellipse with a height of 1 is drawn correctly
        when spanning the width of the surface.

        An ellipse with a height of 1 pixel is a horizontal line.
        """
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surf_w, surf_h = (20, 10)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (surf_w + 2, 1))
        positions = ((-1, -1), (-1, 0), (-1, surf_h // 2), (-1, surf_h - 1), (-1, surf_h))
        for rect_pos in positions:
            surface.fill(surface_color)
            rect.topleft = rect_pos
            self.draw_ellipse(surface, ellipse_color, rect)
            self._check_1_pixel_sized_ellipse(surface, rect, surface_color, ellipse_color)

    def test_ellipse__1_pixel_width_and_height(self):
        """Ensures an ellipse with a width and height of 1 is drawn correctly.

        An ellipse with a width and height of 1 pixel is a single pixel.
        """
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surf_w, surf_h = (10, 10)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (1, 1))
        off_left = -1
        off_right = surf_w
        off_top = -1
        off_bottom = surf_h
        left_edge = 0
        right_edge = surf_w - 1
        top_edge = 0
        bottom_edge = surf_h - 1
        center_x = surf_w // 2
        center_y = surf_h // 2
        positions = ((off_left, off_top), (off_left, top_edge), (off_left, center_y), (off_left, bottom_edge), (off_left, off_bottom), (left_edge, off_top), (left_edge, top_edge), (left_edge, center_y), (left_edge, bottom_edge), (left_edge, off_bottom), (center_x, off_top), (center_x, top_edge), (center_x, center_y), (center_x, bottom_edge), (center_x, off_bottom), (right_edge, off_top), (right_edge, top_edge), (right_edge, center_y), (right_edge, bottom_edge), (right_edge, off_bottom), (off_right, off_top), (off_right, top_edge), (off_right, center_y), (off_right, bottom_edge), (off_right, off_bottom))
        for rect_pos in positions:
            surface.fill(surface_color)
            rect.topleft = rect_pos
            self.draw_ellipse(surface, ellipse_color, rect)
            self._check_1_pixel_sized_ellipse(surface, rect, surface_color, ellipse_color)

    def test_ellipse__bounding_rect(self):
        """Ensures draw ellipse returns the correct bounding rect.

        Tests ellipses on and off the surface and a range of width/thickness
        values.
        """
        ellipse_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        min_width = min_height = 5
        max_width = max_height = 7
        sizes = ((min_width, min_height), (max_width, max_height))
        surface = pygame.Surface((20, 20), 0, 32)
        surf_rect = surface.get_rect()
        big_rect = surf_rect.inflate(min_width * 2 + 1, min_height * 2 + 1)
        for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
            for attr in RECT_POSITION_ATTRIBUTES:
                for width, height in sizes:
                    ellipse_rect = pygame.Rect((0, 0), (width, height))
                    setattr(ellipse_rect, attr, pos)
                    for thickness in (0, 1, 2, 3, min(width, height)):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_ellipse(surface, ellipse_color, ellipse_rect, thickness)
                        expected_rect = create_bounding_rect(surface, surf_color, ellipse_rect.topleft)
                        self.assertEqual(bounding_rect, expected_rect)

    def test_ellipse__surface_clip(self):
        """Ensures draw ellipse respects a surface's clip area.

        Tests drawing the ellipse filled and unfilled.
        """
        surfw = surfh = 30
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for width in (0, 1):
            for center in rect_corners_mids_and_center(clip_rect):
                pos_rect.center = center
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_ellipse(surface, ellipse_color, pos_rect, width)
                expected_pts = get_color_points(surface, ellipse_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_ellipse(surface, ellipse_color, pos_rect, width)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = ellipse_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()