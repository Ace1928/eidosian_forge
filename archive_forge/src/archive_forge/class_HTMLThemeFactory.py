import configparser
import os
import shutil
import tempfile
from os import path
from typing import TYPE_CHECKING, Any, Dict, List
from zipfile import ZipFile
from sphinx import package_dir
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import ensuredir
class HTMLThemeFactory:
    """A factory class for HTML Themes."""

    def __init__(self, app: 'Sphinx') -> None:
        self.app = app
        self.themes = app.registry.html_themes
        self.load_builtin_themes()
        if getattr(app.config, 'html_theme_path', None):
            self.load_additional_themes(app.config.html_theme_path)

    def load_builtin_themes(self) -> None:
        """Load built-in themes."""
        themes = self.find_themes(path.join(package_dir, 'themes'))
        for name, theme in themes.items():
            self.themes[name] = theme

    def load_additional_themes(self, theme_paths: str) -> None:
        """Load additional themes placed at specified directories."""
        for theme_path in theme_paths:
            abs_theme_path = path.abspath(path.join(self.app.confdir, theme_path))
            themes = self.find_themes(abs_theme_path)
            for name, theme in themes.items():
                self.themes[name] = theme

    def load_extra_theme(self, name: str) -> None:
        """Try to load a theme with the specified name."""
        if name == 'alabaster':
            self.load_alabaster_theme()
        else:
            self.load_external_theme(name)

    def load_alabaster_theme(self) -> None:
        """Load alabaster theme."""
        import alabaster
        self.themes['alabaster'] = path.join(alabaster.get_path(), 'alabaster')

    def load_sphinx_rtd_theme(self) -> None:
        """Load sphinx_rtd_theme theme (if installed)."""
        try:
            import sphinx_rtd_theme
            theme_path = sphinx_rtd_theme.get_html_theme_path()
            self.themes['sphinx_rtd_theme'] = path.join(theme_path, 'sphinx_rtd_theme')
        except ImportError:
            pass

    def load_external_theme(self, name: str) -> None:
        """Try to load a theme using entry_points.

        Sphinx refers to ``sphinx_themes`` entry_points.
        """
        theme_entry_points = entry_points(group='sphinx.html_themes')
        try:
            entry_point = theme_entry_points[name]
            self.app.registry.load_extension(self.app, entry_point.module)
            self.app.config.post_init_values()
            return
        except KeyError:
            pass

    def find_themes(self, theme_path: str) -> Dict[str, str]:
        """Search themes from specified directory."""
        themes: Dict[str, str] = {}
        if not path.isdir(theme_path):
            return themes
        for entry in os.listdir(theme_path):
            pathname = path.join(theme_path, entry)
            if path.isfile(pathname) and entry.lower().endswith('.zip'):
                if is_archived_theme(pathname):
                    name = entry[:-4]
                    themes[name] = pathname
                else:
                    logger.warning(__('file %r on theme path is not a valid zipfile or contains no theme'), entry)
            elif path.isfile(path.join(pathname, THEMECONF)):
                themes[entry] = pathname
        return themes

    def create(self, name: str) -> Theme:
        """Create an instance of theme."""
        if name not in self.themes:
            self.load_extra_theme(name)
        if name not in self.themes and name == 'sphinx_rtd_theme':
            logger.warning(__('sphinx_rtd_theme (< 0.3.0) found. It will not be available since Sphinx-6.0'))
            self.load_sphinx_rtd_theme()
        if name not in self.themes:
            raise ThemeError(__('no theme named %r found (missing theme.conf?)') % name)
        return Theme(name, self.themes[name], factory=self)