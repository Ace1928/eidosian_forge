import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
class ImageModuleTest(unittest.TestCase):

    def testLoadIcon(self):
        """see if we can load the pygame icon."""
        f = pygame.pkgdata.getResource('pygame_icon.bmp')
        self.assertEqual(f.mode, 'rb')
        surf = pygame.image.load_basic(f)
        self.assertEqual(surf.get_at((0, 0)), (5, 4, 5, 255))
        self.assertEqual(surf.get_height(), 32)
        self.assertEqual(surf.get_width(), 32)

    def testLoadPNG(self):
        """see if we can load a png with color values in the proper channels."""
        reddish_pixel = (210, 0, 0, 255)
        greenish_pixel = (0, 220, 0, 255)
        bluish_pixel = (0, 0, 230, 255)
        greyish_pixel = (110, 120, 130, 140)
        pixel_array = [reddish_pixel + greenish_pixel, bluish_pixel + greyish_pixel]
        f_descriptor, f_path = tempfile.mkstemp(suffix='.png')
        with os.fdopen(f_descriptor, 'wb') as f:
            w = png.Writer(2, 2, alpha=True)
            w.write(f, pixel_array)
        surf = pygame.image.load(f_path)
        self.assertEqual(surf.get_at((0, 0)), reddish_pixel)
        self.assertEqual(surf.get_at((1, 0)), greenish_pixel)
        self.assertEqual(surf.get_at((0, 1)), bluish_pixel)
        self.assertEqual(surf.get_at((1, 1)), greyish_pixel)
        with open(f_path, 'rb') as f:
            surf = pygame.image.load(f)
        self.assertEqual(surf.get_at((0, 0)), reddish_pixel)
        self.assertEqual(surf.get_at((1, 0)), greenish_pixel)
        self.assertEqual(surf.get_at((0, 1)), bluish_pixel)
        self.assertEqual(surf.get_at((1, 1)), greyish_pixel)
        os.remove(f_path)

    def testLoadJPG(self):
        """to see if we can load a jpg."""
        f = example_path('data/alien1.jpg')
        surf = pygame.image.load(f)
        with open(f, 'rb') as f:
            surf = pygame.image.load(f)

    def testLoadBytesIO(self):
        """to see if we can load images with BytesIO."""
        files = ['data/alien1.png', 'data/alien1.jpg', 'data/alien1.gif', 'data/asprite.bmp']
        for fname in files:
            with self.subTest(fname=fname):
                with open(example_path(fname), 'rb') as f:
                    img_bytes = f.read()
                    img_file = io.BytesIO(img_bytes)
                    image = pygame.image.load(img_file)

    @unittest.skipIf(sdl_image_svg_jpeg_save_bug, 'SDL_image 2.0.5 and older has a big endian bug in jpeg saving')
    def testSaveJPG(self):
        """JPG equivalent to issue #211 - color channel swapping

        Make sure the SDL surface color masks represent the rgb memory format
        required by the JPG library. The masks are machine endian dependent
        """
        from pygame import Color, Rect
        square_len = 16
        sz = (2 * square_len, 2 * square_len)

        def as_rect(square_x, square_y):
            return Rect(square_x * square_len, square_y * square_len, square_len, square_len)
        squares = [(as_rect(0, 0), Color('red')), (as_rect(1, 0), Color('green')), (as_rect(0, 1), Color('blue')), (as_rect(1, 1), Color(255, 128, 64))]
        surf = pygame.Surface(sz, 0, 32)
        for rect, color in squares:
            surf.fill(color, rect)
        f_path = tempfile.mktemp(suffix='.jpg')
        pygame.image.save(surf, f_path)
        jpg_surf = pygame.image.load(f_path)

        def approx(c):
            mask = 252
            return pygame.Color(c.r & mask, c.g & mask, c.b & mask)
        offset = square_len // 2
        for rect, color in squares:
            posn = rect.move((offset, offset)).topleft
            self.assertEqual(approx(jpg_surf.get_at(posn)), approx(color))
        os.remove(f_path)

    def testSavePNG32(self):
        """see if we can save a png with color values in the proper channels."""
        reddish_pixel = (215, 0, 0, 255)
        greenish_pixel = (0, 225, 0, 255)
        bluish_pixel = (0, 0, 235, 255)
        greyish_pixel = (115, 125, 135, 145)
        surf = pygame.Surface((1, 4), pygame.SRCALPHA, 32)
        surf.set_at((0, 0), reddish_pixel)
        surf.set_at((0, 1), greenish_pixel)
        surf.set_at((0, 2), bluish_pixel)
        surf.set_at((0, 3), greyish_pixel)
        f_path = tempfile.mktemp(suffix='.png')
        pygame.image.save(surf, f_path)
        try:
            reader = png.Reader(filename=f_path)
            width, height, pixels, metadata = reader.asRGBA8()
            self.assertEqual(tuple(next(pixels)), reddish_pixel)
            self.assertEqual(tuple(next(pixels)), greenish_pixel)
            self.assertEqual(tuple(next(pixels)), bluish_pixel)
            self.assertEqual(tuple(next(pixels)), greyish_pixel)
        finally:
            if not reader.file.closed:
                reader.file.close()
            del reader
            os.remove(f_path)

    def testSavePNG24(self):
        """see if we can save a png with color values in the proper channels."""
        reddish_pixel = (215, 0, 0)
        greenish_pixel = (0, 225, 0)
        bluish_pixel = (0, 0, 235)
        greyish_pixel = (115, 125, 135)
        surf = pygame.Surface((1, 4), 0, 24)
        surf.set_at((0, 0), reddish_pixel)
        surf.set_at((0, 1), greenish_pixel)
        surf.set_at((0, 2), bluish_pixel)
        surf.set_at((0, 3), greyish_pixel)
        f_path = tempfile.mktemp(suffix='.png')
        pygame.image.save(surf, f_path)
        try:
            reader = png.Reader(filename=f_path)
            width, height, pixels, metadata = reader.asRGB8()
            self.assertEqual(tuple(next(pixels)), reddish_pixel)
            self.assertEqual(tuple(next(pixels)), greenish_pixel)
            self.assertEqual(tuple(next(pixels)), bluish_pixel)
            self.assertEqual(tuple(next(pixels)), greyish_pixel)
        finally:
            if not reader.file.closed:
                reader.file.close()
            del reader
            os.remove(f_path)

    def testSavePNG8(self):
        """see if we can save an 8 bit png correctly"""
        set_pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (170, 146, 170)]
        size = (1, len(set_pixels))
        surf = pygame.Surface(size, depth=8)
        for cnt, pix in enumerate(set_pixels):
            surf.set_at((0, cnt), pix)
        f_path = tempfile.mktemp(suffix='.png')
        pygame.image.save(surf, f_path)
        try:
            reader = png.Reader(filename=f_path)
            width, height, pixels, _ = reader.asRGB8()
            self.assertEqual(size, (width, height))
            self.assertEqual(list(map(tuple, pixels)), set_pixels)
        finally:
            if not reader.file.closed:
                reader.file.close()
            del reader
            os.remove(f_path)

    def testSavePaletteAsPNG8(self):
        """see if we can save a png with color values in the proper channels."""
        pygame.display.init()
        reddish_pixel = (215, 0, 0)
        greenish_pixel = (0, 225, 0)
        bluish_pixel = (0, 0, 235)
        greyish_pixel = (115, 125, 135)
        surf = pygame.Surface((1, 4), 0, 8)
        surf.set_palette_at(0, reddish_pixel)
        surf.set_palette_at(1, greenish_pixel)
        surf.set_palette_at(2, bluish_pixel)
        surf.set_palette_at(3, greyish_pixel)
        f_path = tempfile.mktemp(suffix='.png')
        pygame.image.save(surf, f_path)
        try:
            reader = png.Reader(filename=f_path)
            reader.read()
            palette = reader.palette()
            self.assertEqual(tuple(next(palette)), reddish_pixel)
            self.assertEqual(tuple(next(palette)), greenish_pixel)
            self.assertEqual(tuple(next(palette)), bluish_pixel)
            self.assertEqual(tuple(next(palette)), greyish_pixel)
        finally:
            if not reader.file.closed:
                reader.file.close()
            del reader
            os.remove(f_path)

    def test_save(self):
        s = pygame.Surface((10, 10))
        s.fill((23, 23, 23))
        magic_hex = {}
        magic_hex['jpg'] = [255, 216, 255, 224]
        magic_hex['png'] = [137, 80, 78, 71]
        magic_hex['bmp'] = [66, 77]
        formats = ['jpg', 'png', 'bmp']
        formats = formats + [x.upper() for x in formats]
        for fmt in formats:
            try:
                temp_filename = f'tmpimg.{fmt}'
                pygame.image.save(s, temp_filename)
                with open(temp_filename, 'rb') as handle:
                    self.assertEqual((1, fmt), (test_magic(handle, magic_hex[fmt.lower()]), fmt))
                s2 = pygame.image.load(temp_filename)
                self.assertEqual(s2.get_at((0, 0)), s.get_at((0, 0)))
            finally:
                os.remove(temp_filename)

    def test_save_to_fileobject(self):
        s = pygame.Surface((1, 1))
        s.fill((23, 23, 23))
        bytes_stream = io.BytesIO()
        pygame.image.save(s, bytes_stream)
        bytes_stream.seek(0)
        s2 = pygame.image.load(bytes_stream, 'tga')
        self.assertEqual(s.get_at((0, 0)), s2.get_at((0, 0)))

    def test_save_tga(self):
        s = pygame.Surface((1, 1))
        s.fill((23, 23, 23))
        with tempfile.NamedTemporaryFile(suffix='.tga', delete=False) as f:
            temp_filename = f.name
        try:
            pygame.image.save(s, temp_filename)
            s2 = pygame.image.load(temp_filename)
            self.assertEqual(s2.get_at((0, 0)), s.get_at((0, 0)))
        finally:
            os.remove(temp_filename)

    def test_save_pathlib(self):
        surf = pygame.Surface((1, 1))
        surf.fill((23, 23, 23))
        with tempfile.NamedTemporaryFile(suffix='.tga', delete=False) as f:
            temp_filename = f.name
        path = pathlib.Path(temp_filename)
        try:
            pygame.image.save(surf, path)
            s2 = pygame.image.load(path)
            self.assertEqual(s2.get_at((0, 0)), surf.get_at((0, 0)))
        finally:
            os.remove(temp_filename)

    def test_save__to_fileobject_w_namehint_argument(self):
        s = pygame.Surface((10, 10))
        s.fill((23, 23, 23))
        magic_hex = {}
        magic_hex['jpg'] = [255, 216, 255, 224]
        magic_hex['png'] = [137, 80, 78, 71]
        magic_hex['bmp'] = [66, 77]
        formats = ['tga', 'jpg', 'bmp', 'png']
        formats = formats + [x.upper() for x in formats]
        SDL_Im_version = pygame.image.get_sdl_image_version()
        isAtLeastSDL_image_2_0_2 = SDL_Im_version is not None and SDL_Im_version[0] * 10000 + SDL_Im_version[1] * 100 + SDL_Im_version[2] >= 20002
        for fmt in formats:
            tmp_file, tmp_filename = tempfile.mkstemp(suffix=f'.{fmt}')
            if not isAtLeastSDL_image_2_0_2 and fmt.lower() == 'jpg':
                with os.fdopen(tmp_file, 'wb') as handle:
                    with self.assertRaises(pygame.error):
                        pygame.image.save(s, handle, tmp_filename)
            else:
                with os.fdopen(tmp_file, 'r+b') as handle:
                    pygame.image.save(s, handle, tmp_filename)
                    if fmt.lower() in magic_hex:
                        handle.seek(0)
                        self.assertEqual((1, fmt), (test_magic(handle, magic_hex[fmt.lower()]), fmt))
                    handle.flush()
                    handle.seek(0)
                    s2 = pygame.image.load(handle, tmp_filename)
                    self.assertEqual(s2.get_at((0, 0)), s.get_at((0, 0)))
            os.remove(tmp_filename)

    def test_save_colorkey(self):
        """make sure the color key is not changed when saving."""
        s = pygame.Surface((10, 10), pygame.SRCALPHA, 32)
        s.fill((23, 23, 23))
        s.set_colorkey((0, 0, 0))
        colorkey1 = s.get_colorkey()
        p1 = s.get_at((0, 0))
        temp_filename = 'tmpimg.png'
        try:
            pygame.image.save(s, temp_filename)
            s2 = pygame.image.load(temp_filename)
        finally:
            os.remove(temp_filename)
        colorkey2 = s.get_colorkey()
        self.assertEqual(colorkey1, colorkey2)
        self.assertEqual(p1, s2.get_at((0, 0)))

    def test_load_unicode_path(self):
        import shutil
        orig = example_path('data/asprite.bmp')
        temp = os.path.join(example_path('data'), '你好.bmp')
        shutil.copy(orig, temp)
        try:
            im = pygame.image.load(temp)
        finally:
            os.remove(temp)

    def _unicode_save(self, temp_file):
        im = pygame.Surface((10, 10), 0, 32)
        try:
            with open(temp_file, 'w') as f:
                pass
            os.remove(temp_file)
        except OSError:
            raise unittest.SkipTest('the path cannot be opened')
        self.assertFalse(os.path.exists(temp_file))
        try:
            pygame.image.save(im, temp_file)
            self.assertGreater(os.path.getsize(temp_file), 10)
        finally:
            try:
                os.remove(temp_file)
            except OSError:
                pass

    def test_save_unicode_path(self):
        """save unicode object with non-ASCII chars"""
        self._unicode_save('你好.bmp')

    def assertPremultipliedAreEqual(self, string1, string2, source_string):
        self.assertEqual(len(string1), len(string2))
        block_size = 20
        if string1 != string2:
            for block_start in range(0, len(string1), block_size):
                block_end = min(block_start + block_size, len(string1))
                block1 = string1[block_start:block_end]
                block2 = string2[block_start:block_end]
                if block1 != block2:
                    source_block = source_string[block_start:block_end]
                    msg = 'string difference in %d to %d of %d:\n%s\n%s\nsource:\n%s' % (block_start, block_end, len(string1), binascii.hexlify(block1), binascii.hexlify(block2), binascii.hexlify(source_block))
                    self.fail(msg)

    def test_to_string__premultiplied(self):
        """test to make sure we can export a surface to a premultiplied alpha string"""

        def convertRGBAtoPremultiplied(surface_to_modify):
            for x in range(surface_to_modify.get_width()):
                for y in range(surface_to_modify.get_height()):
                    color = surface_to_modify.get_at((x, y))
                    premult_color = (color[0] * color[3] / 255, color[1] * color[3] / 255, color[2] * color[3] / 255, color[3])
                    surface_to_modify.set_at((x, y), premult_color)
        test_surface = pygame.Surface((256, 256), pygame.SRCALPHA, 32)
        for x in range(test_surface.get_width()):
            for y in range(test_surface.get_height()):
                i = x + y * test_surface.get_width()
                test_surface.set_at((x, y), (i * 7 % 256, i * 13 % 256, i * 27 % 256, y))
        premultiplied_copy = test_surface.copy()
        convertRGBAtoPremultiplied(premultiplied_copy)
        self.assertPremultipliedAreEqual(pygame.image.tostring(test_surface, 'RGBA_PREMULT'), pygame.image.tostring(premultiplied_copy, 'RGBA'), pygame.image.tostring(test_surface, 'RGBA'))
        self.assertPremultipliedAreEqual(pygame.image.tostring(test_surface, 'ARGB_PREMULT'), pygame.image.tostring(premultiplied_copy, 'ARGB'), pygame.image.tostring(test_surface, 'ARGB'))
        no_alpha_surface = pygame.Surface((256, 256), 0, 24)
        self.assertRaises(ValueError, pygame.image.tostring, no_alpha_surface, 'RGBA_PREMULT')

    def _assertSurfaceEqual(self, surf_a, surf_b, msg=None):
        a_width, a_height = (surf_a.get_width(), surf_a.get_height())
        self.assertEqual(a_width, surf_b.get_width(), msg)
        self.assertEqual(a_height, surf_b.get_height(), msg)
        self.assertEqual(surf_a.get_size(), surf_b.get_size(), msg)
        self.assertEqual(surf_a.get_rect(), surf_b.get_rect(), msg)
        self.assertEqual(surf_a.get_colorkey(), surf_b.get_colorkey(), msg)
        self.assertEqual(surf_a.get_alpha(), surf_b.get_alpha(), msg)
        self.assertEqual(surf_a.get_flags(), surf_b.get_flags(), msg)
        self.assertEqual(surf_a.get_bitsize(), surf_b.get_bitsize(), msg)
        self.assertEqual(surf_a.get_bytesize(), surf_b.get_bytesize(), msg)
        surf_a_get_at = surf_a.get_at
        surf_b_get_at = surf_b.get_at
        for y in range(a_height):
            for x in range(a_width):
                self.assertEqual(surf_a_get_at((x, y)), surf_b_get_at((x, y)), '%s (pixel: %d, %d)' % (msg, x, y))

    def test_fromstring__and_tostring(self):
        """Ensure methods tostring() and fromstring() are symmetric."""
        import itertools
        fmts = ('RGBA', 'ARGB', 'BGRA')
        fmt_permutations = itertools.permutations(fmts, 2)
        fmt_combinations = itertools.combinations(fmts, 2)

        def convert(fmt1, fmt2, str_buf):
            pos_fmt1 = {k: v for v, k in enumerate(fmt1)}
            pos_fmt2 = {k: v for v, k in enumerate(fmt2)}
            byte_buf = array.array('B', str_buf)
            num_quads = len(byte_buf) // 4
            for i in range(num_quads):
                i4 = i * 4
                R = byte_buf[i4 + pos_fmt1['R']]
                G = byte_buf[i4 + pos_fmt1['G']]
                B = byte_buf[i4 + pos_fmt1['B']]
                A = byte_buf[i4 + pos_fmt1['A']]
                byte_buf[i4 + pos_fmt2['R']] = R
                byte_buf[i4 + pos_fmt2['G']] = G
                byte_buf[i4 + pos_fmt2['B']] = B
                byte_buf[i4 + pos_fmt2['A']] = A
            return tostring(byte_buf)
        test_surface = pygame.Surface((64, 256), flags=pygame.SRCALPHA, depth=32)
        for i in range(256):
            for j in range(16):
                intensity = j * 16 + 15
                test_surface.set_at((j + 0, i), (intensity, i, i, i))
                test_surface.set_at((j + 16, i), (i, intensity, i, i))
                test_surface.set_at((j + 32, i), (i, i, intensity, i))
                test_surface.set_at((j + 32, i), (i, i, i, intensity))
        self._assertSurfaceEqual(test_surface, test_surface, 'failing with identical surfaces')
        for pair in fmt_combinations:
            fmt1_buf = pygame.image.tostring(test_surface, pair[0])
            fmt1_convert_buf = convert(pair[1], pair[0], convert(pair[0], pair[1], fmt1_buf))
            test_convert_two_way = pygame.image.fromstring(fmt1_convert_buf, test_surface.get_size(), pair[0])
            self._assertSurfaceEqual(test_surface, test_convert_two_way, f'converting {pair[0]} to {pair[1]} and back is not symmetric')
        for pair in fmt_permutations:
            fmt1_buf = pygame.image.tostring(test_surface, pair[0])
            fmt2_convert_buf = convert(pair[0], pair[1], fmt1_buf)
            test_convert_one_way = pygame.image.fromstring(fmt2_convert_buf, test_surface.get_size(), pair[1])
            self._assertSurfaceEqual(test_surface, test_convert_one_way, f'converting {pair[0]} to {pair[1]} failed')
        for fmt in fmts:
            test_buf = pygame.image.tostring(test_surface, fmt)
            test_to_from_fmt_string = pygame.image.fromstring(test_buf, test_surface.get_size(), fmt)
            self._assertSurfaceEqual(test_surface, test_to_from_fmt_string, f"tostring/fromstring functions are not symmetric with '{fmt}' format")

    def test_tostring_depth_24(self):
        test_surface = pygame.Surface((64, 256), depth=24)
        for i in range(256):
            for j in range(16):
                intensity = j * 16 + 15
                test_surface.set_at((j + 0, i), (intensity, i, i, i))
                test_surface.set_at((j + 16, i), (i, intensity, i, i))
                test_surface.set_at((j + 32, i), (i, i, intensity, i))
                test_surface.set_at((j + 32, i), (i, i, i, intensity))
        fmt = 'RGB'
        fmt_buf = pygame.image.tostring(test_surface, fmt)
        test_to_from_fmt_string = pygame.image.fromstring(fmt_buf, test_surface.get_size(), fmt)
        self._assertSurfaceEqual(test_surface, test_to_from_fmt_string, f'tostring/fromstring functions are not symmetric with "{fmt}" format')

    def test_frombuffer_8bit(self):
        """test reading pixel data from a bytes buffer"""
        pygame.display.init()
        eight_bit_palette_buffer = bytearray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        eight_bit_surf = pygame.image.frombuffer(eight_bit_palette_buffer, (4, 4), 'P')
        eight_bit_surf.set_palette([(255, 10, 20), (255, 255, 255), (0, 0, 0), (50, 200, 20)])
        self.assertEqual(eight_bit_surf.get_at((0, 0)), pygame.Color(255, 10, 20))
        self.assertEqual(eight_bit_surf.get_at((1, 1)), pygame.Color(255, 255, 255))
        self.assertEqual(eight_bit_surf.get_at((2, 2)), pygame.Color(0, 0, 0))
        self.assertEqual(eight_bit_surf.get_at((3, 3)), pygame.Color(50, 200, 20))

    def test_frombuffer_RGB(self):
        rgb_buffer = bytearray([255, 10, 20, 255, 10, 20, 255, 10, 20, 255, 10, 20, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 200, 20, 50, 200, 20, 50, 200, 20, 50, 200, 20])
        rgb_surf = pygame.image.frombuffer(rgb_buffer, (4, 4), 'RGB')
        self.assertEqual(rgb_surf.get_at((0, 0)), pygame.Color(255, 10, 20))
        self.assertEqual(rgb_surf.get_at((1, 1)), pygame.Color(255, 255, 255))
        self.assertEqual(rgb_surf.get_at((2, 2)), pygame.Color(0, 0, 0))
        self.assertEqual(rgb_surf.get_at((3, 3)), pygame.Color(50, 200, 20))

    def test_frombuffer_BGR(self):
        bgr_buffer = bytearray([20, 10, 255, 20, 10, 255, 20, 10, 255, 20, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 200, 50, 20, 200, 50, 20, 200, 50, 20, 200, 50])
        bgr_surf = pygame.image.frombuffer(bgr_buffer, (4, 4), 'BGR')
        self.assertEqual(bgr_surf.get_at((0, 0)), pygame.Color(255, 10, 20))
        self.assertEqual(bgr_surf.get_at((1, 1)), pygame.Color(255, 255, 255))
        self.assertEqual(bgr_surf.get_at((2, 2)), pygame.Color(0, 0, 0))
        self.assertEqual(bgr_surf.get_at((3, 3)), pygame.Color(50, 200, 20))

    def test_frombuffer_BGRA(self):
        bgra_buffer = bytearray([255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 127, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 79, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255])
        bgra_surf = pygame.image.frombuffer(bgra_buffer, (4, 4), 'BGRA')
        self.assertEqual(bgra_surf.get_at((0, 0)), pygame.Color(20, 10, 255, 200))
        self.assertEqual(bgra_surf.get_at((1, 1)), pygame.Color(255, 255, 255, 127))
        self.assertEqual(bgra_surf.get_at((2, 2)), pygame.Color(0, 0, 0, 79))
        self.assertEqual(bgra_surf.get_at((3, 3)), pygame.Color(20, 200, 50, 255))

    def test_frombuffer_RGBX(self):
        rgbx_buffer = bytearray([255, 10, 20, 255, 255, 10, 20, 255, 255, 10, 20, 255, 255, 10, 20, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255])
        rgbx_surf = pygame.image.frombuffer(rgbx_buffer, (4, 4), 'RGBX')
        self.assertEqual(rgbx_surf.get_at((0, 0)), pygame.Color(255, 10, 20, 255))
        self.assertEqual(rgbx_surf.get_at((1, 1)), pygame.Color(255, 255, 255, 255))
        self.assertEqual(rgbx_surf.get_at((2, 2)), pygame.Color(0, 0, 0, 255))
        self.assertEqual(rgbx_surf.get_at((3, 3)), pygame.Color(50, 200, 20, 255))

    def test_frombuffer_RGBA(self):
        rgba_buffer = bytearray([255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 127, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 79, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255])
        rgba_surf = pygame.image.frombuffer(rgba_buffer, (4, 4), 'RGBA')
        self.assertEqual(rgba_surf.get_at((0, 0)), pygame.Color(255, 10, 20, 200))
        self.assertEqual(rgba_surf.get_at((1, 1)), pygame.Color(255, 255, 255, 127))
        self.assertEqual(rgba_surf.get_at((2, 2)), pygame.Color(0, 0, 0, 79))
        self.assertEqual(rgba_surf.get_at((3, 3)), pygame.Color(50, 200, 20, 255))

    def test_frombuffer_ARGB(self):
        argb_buffer = bytearray([200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 127, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 79, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20])
        argb_surf = pygame.image.frombuffer(argb_buffer, (4, 4), 'ARGB')
        self.assertEqual(argb_surf.get_at((0, 0)), pygame.Color(255, 10, 20, 200))
        self.assertEqual(argb_surf.get_at((1, 1)), pygame.Color(255, 255, 255, 127))
        self.assertEqual(argb_surf.get_at((2, 2)), pygame.Color(0, 0, 0, 79))
        self.assertEqual(argb_surf.get_at((3, 3)), pygame.Color(50, 200, 20, 255))

    def test_get_extended(self):
        raw_image = []
        raw_image.append((200, 200, 200, 255, 100, 100, 100, 255))
        f_descriptor, f_path = tempfile.mkstemp(suffix='.png')
        with os.fdopen(f_descriptor, 'wb') as file:
            w = png.Writer(2, 1, alpha=True)
            w.write(file, raw_image)
        try:
            surf = pygame.image.load(f_path)
            loaded = True
        except pygame.error:
            loaded = False
        self.assertEqual(pygame.image.get_extended(), loaded)
        os.remove(f_path)

    def test_get_sdl_image_version(self):
        if not pygame.image.get_extended():
            self.assertIsNone(pygame.image.get_sdl_image_version())
        else:
            expected_length = 3
            expected_type = tuple
            expected_item_type = int
            version = pygame.image.get_sdl_image_version()
            self.assertIsInstance(version, expected_type)
            self.assertEqual(len(version), expected_length)
            for item in version:
                self.assertIsInstance(item, expected_item_type)

    def test_load_basic(self):
        """to see if we can load bmp from files and/or file-like objects in memory"""
        s = pygame.image.load_basic(example_path('data/asprite.bmp'))
        self.assertEqual(s.get_at((0, 0)), (255, 255, 255, 255))
        f = pygame.pkgdata.getResource('pygame_icon.bmp')
        self.assertEqual(f.mode, 'rb')
        surf = pygame.image.load_basic(f)
        self.assertEqual(surf.get_at((0, 0)), (5, 4, 5, 255))
        self.assertEqual(surf.get_height(), 32)
        self.assertEqual(surf.get_width(), 32)
        f.close()

    def test_load_extended(self):
        """can load different format images.

        We test loading the following file types:
            bmp, png, jpg, gif (non-animated), pcx, tga (uncompressed), tif, xpm, ppm, pgm
        Following file types are tested when using SDL 2
            svg, pnm, webp
        All the loaded images are smaller than 32 x 32 pixels.
        """
        filename_expected_color = [('asprite.bmp', (255, 255, 255, 255)), ('laplacian.png', (10, 10, 70, 255)), ('red.jpg', (254, 0, 0, 255)), ('blue.gif', (0, 0, 255, 255)), ('green.pcx', (0, 255, 0, 255)), ('yellow.tga', (255, 255, 0, 255)), ('turquoise.tif', (0, 255, 255, 255)), ('purple.xpm', (255, 0, 255, 255)), ('black.ppm', (0, 0, 0, 255)), ('grey.pgm', (120, 120, 120, 255)), ('teal.svg', (0, 128, 128, 255)), ('crimson.pnm', (220, 20, 60, 255)), ('scarlet.webp', (252, 14, 53, 255))]
        for filename, expected_color in filename_expected_color:
            if filename.endswith('svg') and sdl_image_svg_jpeg_save_bug:
                continue
            with self.subTest(f'Test loading a {filename.split('.')[-1]}', filename='examples/data/' + filename, expected_color=expected_color):
                surf = pygame.image.load_extended(example_path('data/' + filename))
                self.assertEqual(surf.get_at((0, 0)), expected_color)

    def test_load_pathlib(self):
        """works loading using a Path argument."""
        path = pathlib.Path(example_path('data/asprite.bmp'))
        surf = pygame.image.load_extended(path)
        self.assertEqual(surf.get_at((0, 0)), (255, 255, 255, 255))

    def test_save_extended(self):
        surf = pygame.Surface((5, 5))
        surf.fill((23, 23, 23))
        passing_formats = ['jpg', 'png']
        passing_formats += [fmt.upper() for fmt in passing_formats]
        magic_hex = {}
        magic_hex['jpg'] = [255, 216, 255, 224]
        magic_hex['png'] = [137, 80, 78, 71]
        failing_formats = ['bmp', 'tga']
        failing_formats += [fmt.upper() for fmt in failing_formats]
        for fmt in passing_formats:
            temp_file_name = f'temp_file.{fmt}'
            pygame.image.save_extended(surf, temp_file_name)
            with open(temp_file_name, 'rb') as file:
                self.assertEqual(1, test_magic(file, magic_hex[fmt.lower()]))
            loaded_file = pygame.image.load(temp_file_name)
            self.assertEqual(loaded_file.get_at((0, 0)), surf.get_at((0, 0)))
            os.remove(temp_file_name)
        for fmt in failing_formats:
            self.assertRaises(pygame.error, pygame.image.save_extended, surf, f'temp_file.{fmt}')

    def threads_load(self, images):
        import pygame.threads
        for i in range(10):
            surfs = pygame.threads.tmap(pygame.image.load, images)
            for s in surfs:
                self.assertIsInstance(s, pygame.Surface)

    def test_load_png_threads(self):
        self.threads_load(glob.glob(example_path('data/*.png')))

    def test_load_jpg_threads(self):
        self.threads_load(glob.glob(example_path('data/*.jpg')))

    def test_load_bmp_threads(self):
        self.threads_load(glob.glob(example_path('data/*.bmp')))

    def test_load_gif_threads(self):
        self.threads_load(glob.glob(example_path('data/*.gif')))

    def test_from_to_bytes_exists(self):
        getattr(pygame.image, 'frombytes')
        getattr(pygame.image, 'tobytes')