import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
class PSSurface(Surface):
    """Creates a PostScript surface of the specified size in PostScript points
    to be written to ``target``.

    Note that the size of individual pages of the PostScript output can vary.
    See :meth:`set_size`.

    ``target`` can be :obj:`None` to specify no output.
    This will generate a surface that may be queried and used as a source,
    without generating a temporary file.

    The PostScript surface backend recognizes the ``image/jpeg`` MIME type
    for the data attached to a surface (see :meth:`~Surface.set_mime_data`)
    when it is used as a source pattern for drawing on this surface.
    If it is specified, the PostScript backend emits an image
    with the content of MIME data (with the ``/DCTDecode`` filter)
    instead of a surface snapshot (with the ``/FlateDecode`` filter),
    which typically produces PostScript with a smaller file size.

    :param target:
        A filename,
        a binary mode :term:`file object` with a ``write`` method,
        or :obj:`None`.
    :param width_in_points:
        Width of the surface, in points (1 point == 1/72.0 inch)
    :param height_in_points:
        Height of the surface, in points (1 point == 1/72.0 inch)
    :type width_in_points: float
    :type height_in_points: float

    """

    def __init__(self, target, width_in_points, height_in_points):
        if hasattr(target, 'write') or target is None:
            write_func = _make_write_func(target)
            pointer = cairo.cairo_ps_surface_create_for_stream(write_func, ffi.NULL, width_in_points, height_in_points)
        else:
            write_func = None
            pointer = cairo.cairo_ps_surface_create(_encode_filename(target), width_in_points, height_in_points)
        Surface.__init__(self, pointer, target_keep_alive=write_func)

    def dsc_comment(self, comment):
        """ Emit a comment into the PostScript output for the given surface.

        The comment is expected to conform to
        the PostScript Language Document Structuring Conventions (DSC).
        Please see that manual for details on the available comments
        and their meanings.
        In particular, the ``%%IncludeFeature`` comment allows
        a device-independent means of controlling printer device features.
        So the PostScript Printer Description Files Specification
        will also be a useful reference.

        The comment string must begin with a percent character (%)
        and the total length of the string
        (including any initial percent characters)
        must not exceed 255 bytes.
        Violating either of these conditions will
        place surface into an error state.
        But beyond these two conditions,
        this method will not enforce conformance of the comment
        with any particular specification.

        The comment string should not have a trailing newline.

        The DSC specifies different sections
        in which particular comments can appear.
        This method provides for comments to be emitted
        within three sections:
        the header, the Setup section, and the PageSetup section.
        Comments appearing in the first two sections
        apply to the entire document
        while comments in the BeginPageSetup section
        apply only to a single page.

        For comments to appear in the header section,
        this method should be called after the surface is created,
        but before a call to :meth:`dsc_begin_setup`.

        For comments to appear in the Setup section,
        this method should be called after a call to :meth:`dsc_begin_setup`
        but before a call to :meth:`dsc_begin_page_setup`.

        For comments to appear in the PageSetup section,
        this method should be called after a call to
        :meth:`dsc_begin_page_setup`.

        Note that it is only necessary to call :meth:`dsc_begin_page_setup`
        for the first page of any surface.
        After a call to :meth:`~Surface.show_page`
        or :meth:`~Surface.copy_page`
        comments are unambiguously directed
        to the PageSetup section of the current page.
        But it doesn't hurt to call this method
        at the beginning of every page
        as that consistency may make the calling code simpler.

        As a final note,
        cairo automatically generates several comments on its own.
        As such, applications must not manually generate
        any of the following comments:

        Header section: ``%!PS-Adobe-3.0``, ``%%Creator``, ``%%CreationDate``,
        ``%%Pages``, ``%%BoundingBox``, ``%%DocumentData``,
        ``%%LanguageLevel``, ``%%EndComments``.

        Setup section: ``%%BeginSetup``, ``%%EndSetup``.

        PageSetup section: ``%%BeginPageSetup``, ``%%PageBoundingBox``,
        ``%%EndPageSetup``.

        Other sections: ``%%BeginProlog``, ``%%EndProlog``, ``%%Page``,
        ``%%Trailer``, ``%%EOF``.

        """
        cairo.cairo_ps_surface_dsc_comment(self._pointer, _encode_string(comment))
        self._check_status()

    def dsc_begin_setup(self):
        """Indicate that subsequent calls to :meth:`dsc_comment` should
        direct comments to the Setup section of the PostScript output.

        This method should be called at most once per surface,
        and must be called before any call to :meth:`dsc_begin_page_setup`
        and before any drawing is performed to the surface.

        See :meth:`dsc_comment` for more details.

        """
        cairo.cairo_ps_surface_dsc_begin_setup(self._pointer)
        self._check_status()

    def dsc_begin_page_setup(self):
        """Indicate that subsequent calls to :meth:`dsc_comment` should
        direct comments to the PageSetup section of the PostScript output.

        This method is only needed for the first page of a surface.
        It must be called after any call to :meth:`dsc_begin_setup`
        and before any drawing is performed to the surface.

        See :meth:`dsc_comment` for more details.

        """
        cairo.cairo_ps_surface_dsc_begin_page_setup(self._pointer)
        self._check_status()

    def set_eps(self, eps):
        """
        If ``eps`` is True,
        the PostScript surface will output Encapsulated PostScript.

        This method should only be called
        before any drawing operations have been performed on the current page.
        The simplest way to do this is to call this method
        immediately after creating the surface.
        An Encapsulated PostScript file should never contain
        more than one page.

        """
        cairo.cairo_ps_surface_set_eps(self._pointer, bool(eps))
        self._check_status()

    def get_eps(self):
        """Check whether the PostScript surface will output
        Encapsulated PostScript.

        """
        return bool(cairo.cairo_ps_surface_get_eps(self._pointer))

    def set_size(self, width_in_points, height_in_points):
        """Changes the size of a PostScript surface
        for the current (and subsequent) pages.

        This method should only be called
        before any drawing operations have been performed on the current page.
        The simplest way to do this is to call this method
        immediately after creating the surface
        or immediately after completing a page with either
        :meth:`~Surface.show_page` or :meth:`~Surface.copy_page`.

        :param width_in_points:
            New width of the page, in points (1 point == 1/72.0 inch)
        :param height_in_points:
            New height of the page, in points (1 point == 1/72.0 inch)
        :type width_in_points: float
        :type height_in_points: float

        """
        cairo.cairo_ps_surface_set_size(self._pointer, width_in_points, height_in_points)
        self._check_status()

    def restrict_to_level(self, level):
        """Restricts the generated PostScript file to ``level``.

        See :meth:`get_levels` for a list of available level values
        that can be used here.

        This method should only be called
        before any drawing operations have been performed on the given surface.
        The simplest way to do this is to call this method
        immediately after creating the surface.

        :param version: A :ref:`PS_LEVEL` string.

        """
        cairo.cairo_ps_surface_restrict_to_level(self._pointer, level)
        self._check_status()

    @staticmethod
    def get_levels():
        """Return the list of supported PostScript levels.
        See :meth:`restrict_to_level`.

        :return: A list of :ref:`PS_LEVEL` strings.

        """
        levels = ffi.new('cairo_ps_level_t const **')
        num_levels = ffi.new('int *')
        cairo.cairo_ps_get_levels(levels, num_levels)
        levels = levels[0]
        return [levels[i] for i in range(num_levels[0])]

    @staticmethod
    def ps_level_to_string(level):
        """Return the string representation of the given :ref:`PS_LEVEL`.
        See :meth:`get_levels` for a way to get
        the list of valid level ids.

        """
        c_string = cairo.cairo_ps_level_to_string(level)
        if c_string == ffi.NULL:
            raise ValueError(level)
        return ffi.string(c_string).decode('ascii')