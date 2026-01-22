from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
class ImageDraw:
    font = None

    def __init__(self, im, mode=None):
        """
        Create a drawing instance.

        :param im: The image to draw in.
        :param mode: Optional mode to use for color values.  For RGB
           images, this argument can be RGB or RGBA (to blend the
           drawing into the image).  For all other modes, this argument
           must be the same as the image mode.  If omitted, the mode
           defaults to the mode of the image.
        """
        im.load()
        if im.readonly:
            im._copy()
        blend = 0
        if mode is None:
            mode = im.mode
        if mode != im.mode:
            if mode == 'RGBA' and im.mode == 'RGB':
                blend = 1
            else:
                msg = 'mode mismatch'
                raise ValueError(msg)
        if mode == 'P':
            self.palette = im.palette
        else:
            self.palette = None
        self._image = im
        self.im = im.im
        self.draw = Image.core.draw(self.im, blend)
        self.mode = mode
        if mode in ('I', 'F'):
            self.ink = self.draw.draw_ink(1)
        else:
            self.ink = self.draw.draw_ink(-1)
        if mode in ('1', 'P', 'I', 'F'):
            self.fontmode = '1'
        else:
            self.fontmode = 'L'
        self.fill = False

    def getfont(self):
        """
        Get the current default font.

        To set the default font for this ImageDraw instance::

            from PIL import ImageDraw, ImageFont
            draw.font = ImageFont.truetype("Tests/fonts/FreeMono.ttf")

        To set the default font for all future ImageDraw instances::

            from PIL import ImageDraw, ImageFont
            ImageDraw.ImageDraw.font = ImageFont.truetype("Tests/fonts/FreeMono.ttf")

        If the current default font is ``None``,
        it is initialized with ``ImageFont.load_default()``.

        :returns: An image font."""
        if not self.font:
            from . import ImageFont
            self.font = ImageFont.load_default()
        return self.font

    def _getfont(self, font_size):
        if font_size is not None:
            from . import ImageFont
            font = ImageFont.load_default(font_size)
        else:
            font = self.getfont()
        return font

    def _getink(self, ink, fill=None):
        if ink is None and fill is None:
            if self.fill:
                fill = self.ink
            else:
                ink = self.ink
        else:
            if ink is not None:
                if isinstance(ink, str):
                    ink = ImageColor.getcolor(ink, self.mode)
                if self.palette and (not isinstance(ink, numbers.Number)):
                    ink = self.palette.getcolor(ink, self._image)
                ink = self.draw.draw_ink(ink)
            if fill is not None:
                if isinstance(fill, str):
                    fill = ImageColor.getcolor(fill, self.mode)
                if self.palette and (not isinstance(fill, numbers.Number)):
                    fill = self.palette.getcolor(fill, self._image)
                fill = self.draw.draw_ink(fill)
        return (ink, fill)

    def arc(self, xy, start, end, fill=None, width=1):
        """Draw an arc."""
        ink, fill = self._getink(fill)
        if ink is not None:
            self.draw.draw_arc(xy, start, end, ink, width)

    def bitmap(self, xy, bitmap, fill=None):
        """Draw a bitmap."""
        bitmap.load()
        ink, fill = self._getink(fill)
        if ink is None:
            ink = fill
        if ink is not None:
            self.draw.draw_bitmap(xy, bitmap.im, ink)

    def chord(self, xy, start, end, fill=None, outline=None, width=1):
        """Draw a chord."""
        ink, fill = self._getink(outline, fill)
        if fill is not None:
            self.draw.draw_chord(xy, start, end, fill, 1)
        if ink is not None and ink != fill and (width != 0):
            self.draw.draw_chord(xy, start, end, ink, 0, width)

    def ellipse(self, xy, fill=None, outline=None, width=1):
        """Draw an ellipse."""
        ink, fill = self._getink(outline, fill)
        if fill is not None:
            self.draw.draw_ellipse(xy, fill, 1)
        if ink is not None and ink != fill and (width != 0):
            self.draw.draw_ellipse(xy, ink, 0, width)

    def line(self, xy, fill=None, width=0, joint=None):
        """Draw a line, or a connected sequence of line segments."""
        ink = self._getink(fill)[0]
        if ink is not None:
            self.draw.draw_lines(xy, ink, width)
            if joint == 'curve' and width > 4:
                if not isinstance(xy[0], (list, tuple)):
                    xy = [tuple(xy[i:i + 2]) for i in range(0, len(xy), 2)]
                for i in range(1, len(xy) - 1):
                    point = xy[i]
                    angles = [math.degrees(math.atan2(end[0] - start[0], start[1] - end[1])) % 360 for start, end in ((xy[i - 1], point), (point, xy[i + 1]))]
                    if angles[0] == angles[1]:
                        continue

                    def coord_at_angle(coord, angle):
                        x, y = coord
                        angle -= 90
                        distance = width / 2 - 1
                        return tuple((p + (math.floor(p_d) if p_d > 0 else math.ceil(p_d)) for p, p_d in ((x, distance * math.cos(math.radians(angle))), (y, distance * math.sin(math.radians(angle))))))
                    flipped = angles[1] > angles[0] and angles[1] - 180 > angles[0] or (angles[1] < angles[0] and angles[1] + 180 > angles[0])
                    coords = [(point[0] - width / 2 + 1, point[1] - width / 2 + 1), (point[0] + width / 2 - 1, point[1] + width / 2 - 1)]
                    if flipped:
                        start, end = (angles[1] + 90, angles[0] + 90)
                    else:
                        start, end = (angles[0] - 90, angles[1] - 90)
                    self.pieslice(coords, start - 90, end - 90, fill)
                    if width > 8:
                        if flipped:
                            gap_coords = [coord_at_angle(point, angles[0] + 90), point, coord_at_angle(point, angles[1] + 90)]
                        else:
                            gap_coords = [coord_at_angle(point, angles[0] - 90), point, coord_at_angle(point, angles[1] - 90)]
                        self.line(gap_coords, fill, width=3)

    def shape(self, shape, fill=None, outline=None):
        """(Experimental) Draw a shape."""
        shape.close()
        ink, fill = self._getink(outline, fill)
        if fill is not None:
            self.draw.draw_outline(shape, fill, 1)
        if ink is not None and ink != fill:
            self.draw.draw_outline(shape, ink, 0)

    def pieslice(self, xy, start, end, fill=None, outline=None, width=1):
        """Draw a pieslice."""
        ink, fill = self._getink(outline, fill)
        if fill is not None:
            self.draw.draw_pieslice(xy, start, end, fill, 1)
        if ink is not None and ink != fill and (width != 0):
            self.draw.draw_pieslice(xy, start, end, ink, 0, width)

    def point(self, xy, fill=None):
        """Draw one or more individual pixels."""
        ink, fill = self._getink(fill)
        if ink is not None:
            self.draw.draw_points(xy, ink)

    def polygon(self, xy, fill=None, outline=None, width=1):
        """Draw a polygon."""
        ink, fill = self._getink(outline, fill)
        if fill is not None:
            self.draw.draw_polygon(xy, fill, 1)
        if ink is not None and ink != fill and (width != 0):
            if width == 1:
                self.draw.draw_polygon(xy, ink, 0, width)
            else:
                mask = Image.new('1', self.im.size)
                mask_ink = self._getink(1)[0]
                fill_im = mask.copy()
                draw = Draw(fill_im)
                draw.draw.draw_polygon(xy, mask_ink, 1)
                ink_im = mask.copy()
                draw = Draw(ink_im)
                width = width * 2 - 1
                draw.draw.draw_polygon(xy, mask_ink, 0, width)
                mask.paste(ink_im, mask=fill_im)
                im = Image.new(self.mode, self.im.size)
                draw = Draw(im)
                draw.draw.draw_polygon(xy, ink, 0, width)
                self.im.paste(im.im, (0, 0) + im.size, mask.im)

    def regular_polygon(self, bounding_circle, n_sides, rotation=0, fill=None, outline=None, width=1):
        """Draw a regular polygon."""
        xy = _compute_regular_polygon_vertices(bounding_circle, n_sides, rotation)
        self.polygon(xy, fill, outline, width)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Draw a rectangle."""
        ink, fill = self._getink(outline, fill)
        if fill is not None:
            self.draw.draw_rectangle(xy, fill, 1)
        if ink is not None and ink != fill and (width != 0):
            self.draw.draw_rectangle(xy, ink, 0, width)

    def rounded_rectangle(self, xy, radius=0, fill=None, outline=None, width=1, *, corners=None):
        """Draw a rounded rectangle."""
        if isinstance(xy[0], (list, tuple)):
            (x0, y0), (x1, y1) = xy
        else:
            x0, y0, x1, y1 = xy
        if x1 < x0:
            msg = 'x1 must be greater than or equal to x0'
            raise ValueError(msg)
        if y1 < y0:
            msg = 'y1 must be greater than or equal to y0'
            raise ValueError(msg)
        if corners is None:
            corners = (True, True, True, True)
        d = radius * 2
        full_x, full_y = (False, False)
        if all(corners):
            full_x = d >= x1 - x0 - 1
            if full_x:
                d = x1 - x0
            full_y = d >= y1 - y0 - 1
            if full_y:
                d = y1 - y0
            if full_x and full_y:
                return self.ellipse(xy, fill, outline, width)
        if d == 0 or not any(corners):
            return self.rectangle(xy, fill, outline, width)
        r = d // 2
        ink, fill = self._getink(outline, fill)

        def draw_corners(pieslice):
            if full_x:
                parts = (((x0, y0, x0 + d, y0 + d), 180, 360), ((x0, y1 - d, x0 + d, y1), 0, 180))
            elif full_y:
                parts = (((x0, y0, x0 + d, y0 + d), 90, 270), ((x1 - d, y0, x1, y0 + d), 270, 90))
            else:
                parts = []
                for i, part in enumerate((((x0, y0, x0 + d, y0 + d), 180, 270), ((x1 - d, y0, x1, y0 + d), 270, 360), ((x1 - d, y1 - d, x1, y1), 0, 90), ((x0, y1 - d, x0 + d, y1), 90, 180))):
                    if corners[i]:
                        parts.append(part)
            for part in parts:
                if pieslice:
                    self.draw.draw_pieslice(*part + (fill, 1))
                else:
                    self.draw.draw_arc(*part + (ink, width))
        if fill is not None:
            draw_corners(True)
            if full_x:
                self.draw.draw_rectangle((x0, y0 + r + 1, x1, y1 - r - 1), fill, 1)
            else:
                self.draw.draw_rectangle((x0 + r + 1, y0, x1 - r - 1, y1), fill, 1)
            if not full_x and (not full_y):
                left = [x0, y0, x0 + r, y1]
                if corners[0]:
                    left[1] += r + 1
                if corners[3]:
                    left[3] -= r + 1
                self.draw.draw_rectangle(left, fill, 1)
                right = [x1 - r, y0, x1, y1]
                if corners[1]:
                    right[1] += r + 1
                if corners[2]:
                    right[3] -= r + 1
                self.draw.draw_rectangle(right, fill, 1)
        if ink is not None and ink != fill and (width != 0):
            draw_corners(False)
            if not full_x:
                top = [x0, y0, x1, y0 + width - 1]
                if corners[0]:
                    top[0] += r + 1
                if corners[1]:
                    top[2] -= r + 1
                self.draw.draw_rectangle(top, ink, 1)
                bottom = [x0, y1 - width + 1, x1, y1]
                if corners[3]:
                    bottom[0] += r + 1
                if corners[2]:
                    bottom[2] -= r + 1
                self.draw.draw_rectangle(bottom, ink, 1)
            if not full_y:
                left = [x0, y0, x0 + width - 1, y1]
                if corners[0]:
                    left[1] += r + 1
                if corners[3]:
                    left[3] -= r + 1
                self.draw.draw_rectangle(left, ink, 1)
                right = [x1 - width + 1, y0, x1, y1]
                if corners[1]:
                    right[1] += r + 1
                if corners[2]:
                    right[3] -= r + 1
                self.draw.draw_rectangle(right, ink, 1)

    def _multiline_check(self, text):
        split_character = '\n' if isinstance(text, str) else b'\n'
        return split_character in text

    def _multiline_split(self, text):
        split_character = '\n' if isinstance(text, str) else b'\n'
        return text.split(split_character)

    def _multiline_spacing(self, font, spacing, stroke_width):
        return self.textbbox((0, 0), 'A', font, stroke_width=stroke_width)[3] + stroke_width + spacing

    def text(self, xy, text, fill=None, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, stroke_fill=None, embedded_color=False, *args, **kwargs):
        """Draw text."""
        if embedded_color and self.mode not in ('RGB', 'RGBA'):
            msg = 'Embedded color supported only in RGB and RGBA modes'
            raise ValueError(msg)
        if font is None:
            font = self._getfont(kwargs.get('font_size'))
        if self._multiline_check(text):
            return self.multiline_text(xy, text, fill, font, anchor, spacing, align, direction, features, language, stroke_width, stroke_fill, embedded_color)

        def getink(fill):
            ink, fill = self._getink(fill)
            if ink is None:
                return fill
            return ink

        def draw_text(ink, stroke_width=0, stroke_offset=None):
            mode = self.fontmode
            if stroke_width == 0 and embedded_color:
                mode = 'RGBA'
            coord = []
            start = []
            for i in range(2):
                coord.append(int(xy[i]))
                start.append(math.modf(xy[i])[0])
            try:
                mask, offset = font.getmask2(text, mode, *args, direction=direction, features=features, language=language, stroke_width=stroke_width, anchor=anchor, ink=ink, start=start, **kwargs)
                coord = (coord[0] + offset[0], coord[1] + offset[1])
            except AttributeError:
                try:
                    mask = font.getmask(text, mode, direction, features, language, stroke_width, anchor, ink, *args, start=start, **kwargs)
                except TypeError:
                    mask = font.getmask(text)
            if stroke_offset:
                coord = (coord[0] + stroke_offset[0], coord[1] + stroke_offset[1])
            if mode == 'RGBA':
                color, mask = (mask, mask.getband(3))
                ink_alpha = struct.pack('i', ink)[3]
                color.fillband(3, ink_alpha)
                x, y = coord
                self.im.paste(color, (x, y, x + mask.size[0], y + mask.size[1]), mask)
            else:
                self.draw.draw_bitmap(coord, mask, ink)
        ink = getink(fill)
        if ink is not None:
            stroke_ink = None
            if stroke_width:
                stroke_ink = getink(stroke_fill) if stroke_fill is not None else ink
            if stroke_ink is not None:
                draw_text(stroke_ink, stroke_width)
                draw_text(ink, 0)
            else:
                draw_text(ink)

    def multiline_text(self, xy, text, fill=None, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, stroke_fill=None, embedded_color=False, *, font_size=None):
        if direction == 'ttb':
            msg = 'ttb direction is unsupported for multiline text'
            raise ValueError(msg)
        if anchor is None:
            anchor = 'la'
        elif len(anchor) != 2:
            msg = 'anchor must be a 2 character string'
            raise ValueError(msg)
        elif anchor[1] in 'tb':
            msg = 'anchor not supported for multiline text'
            raise ValueError(msg)
        if font is None:
            font = self._getfont(font_size)
        widths = []
        max_width = 0
        lines = self._multiline_split(text)
        line_spacing = self._multiline_spacing(font, spacing, stroke_width)
        for line in lines:
            line_width = self.textlength(line, font, direction=direction, features=features, language=language)
            widths.append(line_width)
            max_width = max(max_width, line_width)
        top = xy[1]
        if anchor[1] == 'm':
            top -= (len(lines) - 1) * line_spacing / 2.0
        elif anchor[1] == 'd':
            top -= (len(lines) - 1) * line_spacing
        for idx, line in enumerate(lines):
            left = xy[0]
            width_difference = max_width - widths[idx]
            if anchor[0] == 'm':
                left -= width_difference / 2.0
            elif anchor[0] == 'r':
                left -= width_difference
            if align == 'left':
                pass
            elif align == 'center':
                left += width_difference / 2.0
            elif align == 'right':
                left += width_difference
            else:
                msg = 'align must be "left", "center" or "right"'
                raise ValueError(msg)
            self.text((left, top), line, fill, font, anchor, direction=direction, features=features, language=language, stroke_width=stroke_width, stroke_fill=stroke_fill, embedded_color=embedded_color)
            top += line_spacing

    def textlength(self, text, font=None, direction=None, features=None, language=None, embedded_color=False, *, font_size=None):
        """Get the length of a given string, in pixels with 1/64 precision."""
        if self._multiline_check(text):
            msg = "can't measure length of multiline text"
            raise ValueError(msg)
        if embedded_color and self.mode not in ('RGB', 'RGBA'):
            msg = 'Embedded color supported only in RGB and RGBA modes'
            raise ValueError(msg)
        if font is None:
            font = self._getfont(font_size)
        mode = 'RGBA' if embedded_color else self.fontmode
        return font.getlength(text, mode, direction, features, language)

    def textbbox(self, xy, text, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, embedded_color=False, *, font_size=None):
        """Get the bounding box of a given string, in pixels."""
        if embedded_color and self.mode not in ('RGB', 'RGBA'):
            msg = 'Embedded color supported only in RGB and RGBA modes'
            raise ValueError(msg)
        if font is None:
            font = self._getfont(font_size)
        if self._multiline_check(text):
            return self.multiline_textbbox(xy, text, font, anchor, spacing, align, direction, features, language, stroke_width, embedded_color)
        mode = 'RGBA' if embedded_color else self.fontmode
        bbox = font.getbbox(text, mode, direction, features, language, stroke_width, anchor)
        return (bbox[0] + xy[0], bbox[1] + xy[1], bbox[2] + xy[0], bbox[3] + xy[1])

    def multiline_textbbox(self, xy, text, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, embedded_color=False, *, font_size=None):
        if direction == 'ttb':
            msg = 'ttb direction is unsupported for multiline text'
            raise ValueError(msg)
        if anchor is None:
            anchor = 'la'
        elif len(anchor) != 2:
            msg = 'anchor must be a 2 character string'
            raise ValueError(msg)
        elif anchor[1] in 'tb':
            msg = 'anchor not supported for multiline text'
            raise ValueError(msg)
        if font is None:
            font = self._getfont(font_size)
        widths = []
        max_width = 0
        lines = self._multiline_split(text)
        line_spacing = self._multiline_spacing(font, spacing, stroke_width)
        for line in lines:
            line_width = self.textlength(line, font, direction=direction, features=features, language=language, embedded_color=embedded_color)
            widths.append(line_width)
            max_width = max(max_width, line_width)
        top = xy[1]
        if anchor[1] == 'm':
            top -= (len(lines) - 1) * line_spacing / 2.0
        elif anchor[1] == 'd':
            top -= (len(lines) - 1) * line_spacing
        bbox = None
        for idx, line in enumerate(lines):
            left = xy[0]
            width_difference = max_width - widths[idx]
            if anchor[0] == 'm':
                left -= width_difference / 2.0
            elif anchor[0] == 'r':
                left -= width_difference
            if align == 'left':
                pass
            elif align == 'center':
                left += width_difference / 2.0
            elif align == 'right':
                left += width_difference
            else:
                msg = 'align must be "left", "center" or "right"'
                raise ValueError(msg)
            bbox_line = self.textbbox((left, top), line, font, anchor, direction=direction, features=features, language=language, stroke_width=stroke_width, embedded_color=embedded_color)
            if bbox is None:
                bbox = bbox_line
            else:
                bbox = (min(bbox[0], bbox_line[0]), min(bbox[1], bbox_line[1]), max(bbox[2], bbox_line[2]), max(bbox[3], bbox_line[3]))
            top += line_spacing
        if bbox is None:
            return (xy[0], xy[1], xy[0], xy[1])
        return bbox