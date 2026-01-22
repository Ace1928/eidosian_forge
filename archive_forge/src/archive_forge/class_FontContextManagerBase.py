import re
import os
from ast import literal_eval
from functools import partial
from copy import copy
from kivy import kivy_data_dir
from kivy.config import Config
from kivy.utils import platform
from kivy.graphics.texture import Texture
from kivy.core import core_select_lib
from kivy.core.text.text_layout import layout_text, LayoutWord
from kivy.resources import resource_find, resource_add_path
from kivy.compat import PY2
from kivy.setupconfig import USE_SDL2, USE_PANGOFT2
from kivy.logger import Logger
class FontContextManagerBase(object):

    @staticmethod
    def create(font_context):
        """Create a font context, you must specify a unique name (string).
        Returns `True` on success and `False` on failure.

        If `font_context` starts with one of the reserved words `'system://'`,
        `'directory://'`, `'fontconfig://'` or `'systemconfig://'`, the context
        is setup accordingly (exact results of this depends on your platform,
        environment and configuration).

        * `'system://'` loads the default system's FontConfig configuration
          and all fonts (usually including user fonts).
        * `directory://` contexts preload a directory of font files (specified
          in the context name), `systemconfig://` loads the system's FontConfig
          configuration (but no fonts), and `fontconfig://` loads FontConfig
          configuration file (specified in the context name!). These are for
          advanced users only, check the source code and FontConfig
          documentation for details.
        * Fonts automatically loaded to an isolated context (ie when no
          font context was specified) start with `'isolated://'`. This has
          no special effect, and only serves to help you identify them in
          the results returned from :meth:`list`.
        * Any other string is a context that will only draw with the font
          file(s) you explicitly add to it.

        .. versionadded:: 1.11.0

        .. note::
            Font contexts are created automatically by specifying a name in the
            `font_context` property of :class:`kivy.uix.label.Label` or
            :class:`kivy.uix.textinput.TextInput`. They are also auto-created
            by :meth:`add_font` by default, so you normally don't need to
            call this directly.

        .. note:: This feature requires the Pango text provider.
        """
        raise NotImplementedError('No font_context support in text provider')

    @staticmethod
    def exists(font_context):
        """Returns True if a font context with the given name exists.

        .. versionadded:: 1.11.0

        .. note:: This feature requires the Pango text provider.
        """
        raise NotImplementedError('No font_context support in text provider')

    @staticmethod
    def destroy(font_context):
        """Destroy a named font context (if it exists)

        .. versionadded:: 1.11.0

        .. note:: This feature requires the Pango text provider.
        """
        raise NotImplementedError('No font_context support in text provider')

    @staticmethod
    def list():
        """Returns a list of `bytes` objects, each representing a cached font
        context name. Note that entries that start with `isolated://` were
        autocreated by loading a font file with no font_context specified.

        .. versionadded:: 1.11.0

        .. note:: This feature requires the Pango text provider.
        """
        raise NotImplementedError('No font_context support in text provider')

    @staticmethod
    def list_families(font_context):
        """Returns a list of `bytes` objects, each representing a font family
        name that is available in the given `font_context`.

        .. versionadded:: 1.11.0

        .. note::
            Pango adds static "Serif", "Sans" and "Monospace" to the list in
            current versions, even if only a single custom font file is added
            to the context.

        .. note:: This feature requires the Pango text provider.
        """
        raise NotImplementedError('No font_context support in text provider')

    @staticmethod
    def list_custom(font_context):
        """Returns a dictionary representing all the custom-loaded fonts in
        the context. The key is a `bytes` object representing the full path
        to the font file, the value is a `bytes` object representing the font
        family name used to request drawing with the font.

        .. versionadded:: 1.11.0

        .. note:: This feature requires the Pango text provider.
        """
        raise NotImplementedError('No font_context support in text provider')

    @staticmethod
    def add_font(font_context, filename, autocreate=True, family=None):
        """Add a font file to a named font context. If `autocreate` is true,
        the context will be created if it does not exist (this is the
        default). You can specify the `family` argument (string) to skip
        auto-detecting the font family name.

        .. warning::

            The `family` argument is slated for removal if the underlying
            implementation can be fixed, It is offered as a way to optimize
            startup time for deployed applications (it avoids opening the
            file with FreeType2 to determine its family name). To use this,
            first load the font file without specifying `family`, and
            hardcode the returned (autodetected) `family` value in your font
            context initialization.

        .. versionadded:: 1.11.0

        .. note:: This feature requires the Pango text provider.
        """
        raise NotImplementedError('No font_context support in text provider')