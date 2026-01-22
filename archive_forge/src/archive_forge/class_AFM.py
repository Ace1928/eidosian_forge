from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
class AFM:

    def __init__(self, fh):
        """Parse the AFM file in file object *fh*."""
        self._header = _parse_header(fh)
        self._metrics, self._metrics_by_name = _parse_char_metrics(fh)
        self._kern, self._composite = _parse_optional(fh)

    def get_bbox_char(self, c, isord=False):
        if not isord:
            c = ord(c)
        return self._metrics[c].bbox

    def string_width_height(self, s):
        """
        Return the string width (including kerning) and string height
        as a (*w*, *h*) tuple.
        """
        if not len(s):
            return (0, 0)
        total_width = 0
        namelast = None
        miny = 1000000000.0
        maxy = 0
        for c in s:
            if c == '\n':
                continue
            wx, name, bbox = self._metrics[ord(c)]
            total_width += wx + self._kern.get((namelast, name), 0)
            l, b, w, h = bbox
            miny = min(miny, b)
            maxy = max(maxy, b + h)
            namelast = name
        return (total_width, maxy - miny)

    def get_str_bbox_and_descent(self, s):
        """Return the string bounding box and the maximal descent."""
        if not len(s):
            return (0, 0, 0, 0, 0)
        total_width = 0
        namelast = None
        miny = 1000000000.0
        maxy = 0
        left = 0
        if not isinstance(s, str):
            s = _to_str(s)
        for c in s:
            if c == '\n':
                continue
            name = uni2type1.get(ord(c), f'uni{ord(c):04X}')
            try:
                wx, _, bbox = self._metrics_by_name[name]
            except KeyError:
                name = 'question'
                wx, _, bbox = self._metrics_by_name[name]
            total_width += wx + self._kern.get((namelast, name), 0)
            l, b, w, h = bbox
            left = min(left, l)
            miny = min(miny, b)
            maxy = max(maxy, b + h)
            namelast = name
        return (left, miny, total_width, maxy - miny, -miny)

    def get_str_bbox(self, s):
        """Return the string bounding box."""
        return self.get_str_bbox_and_descent(s)[:4]

    def get_name_char(self, c, isord=False):
        """Get the name of the character, i.e., ';' is 'semicolon'."""
        if not isord:
            c = ord(c)
        return self._metrics[c].name

    def get_width_char(self, c, isord=False):
        """
        Get the width of the character from the character metric WX field.
        """
        if not isord:
            c = ord(c)
        return self._metrics[c].width

    def get_width_from_char_name(self, name):
        """Get the width of the character from a type1 character name."""
        return self._metrics_by_name[name].width

    def get_height_char(self, c, isord=False):
        """Get the bounding box (ink) height of character *c* (space is 0)."""
        if not isord:
            c = ord(c)
        return self._metrics[c].bbox[-1]

    def get_kern_dist(self, c1, c2):
        """
        Return the kerning pair distance (possibly 0) for chars *c1* and *c2*.
        """
        name1, name2 = (self.get_name_char(c1), self.get_name_char(c2))
        return self.get_kern_dist_from_name(name1, name2)

    def get_kern_dist_from_name(self, name1, name2):
        """
        Return the kerning pair distance (possibly 0) for chars
        *name1* and *name2*.
        """
        return self._kern.get((name1, name2), 0)

    def get_fontname(self):
        """Return the font name, e.g., 'Times-Roman'."""
        return self._header[b'FontName']

    @property
    def postscript_name(self):
        return self.get_fontname()

    def get_fullname(self):
        """Return the font full name, e.g., 'Times-Roman'."""
        name = self._header.get(b'FullName')
        if name is None:
            name = self._header[b'FontName']
        return name

    def get_familyname(self):
        """Return the font family name, e.g., 'Times'."""
        name = self._header.get(b'FamilyName')
        if name is not None:
            return name
        name = self.get_fullname()
        extras = '(?i)([ -](regular|plain|italic|oblique|bold|semibold|light|ultralight|extra|condensed))+$'
        return re.sub(extras, '', name)

    @property
    def family_name(self):
        """The font family name, e.g., 'Times'."""
        return self.get_familyname()

    def get_weight(self):
        """Return the font weight, e.g., 'Bold' or 'Roman'."""
        return self._header[b'Weight']

    def get_angle(self):
        """Return the fontangle as float."""
        return self._header[b'ItalicAngle']

    def get_capheight(self):
        """Return the cap height as float."""
        return self._header[b'CapHeight']

    def get_xheight(self):
        """Return the xheight as float."""
        return self._header[b'XHeight']

    def get_underline_thickness(self):
        """Return the underline thickness as float."""
        return self._header[b'UnderlineThickness']

    def get_horizontal_stem_width(self):
        """
        Return the standard horizontal stem width as float, or *None* if
        not specified in AFM file.
        """
        return self._header.get(b'StdHW', None)

    def get_vertical_stem_width(self):
        """
        Return the standard vertical stem width as float, or *None* if
        not specified in AFM file.
        """
        return self._header.get(b'StdVW', None)