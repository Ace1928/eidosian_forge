import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class NumpydocParser:
    """Parser for numpydoc-style docstrings."""

    def __init__(self, sections: T.Optional[T.Dict[str, Section]]=None):
        """Setup sections.

        :param sections: Recognized sections or None to defaults.
        """
        sections = sections or DEFAULT_SECTIONS
        self.sections = {s.title: s for s in sections}
        self._setup()

    def _setup(self):
        self.titles_re = re.compile('|'.join((s.title_pattern for s in self.sections.values())), flags=re.M)

    def add_section(self, section: Section):
        """Add or replace a section.

        :param section: The new section.
        """
        self.sections[section.title] = section
        self._setup()

    def parse(self, text: str) -> Docstring:
        """Parse the numpy-style docstring into its components.

        :returns: parsed docstring
        """
        ret = Docstring(style=DocstringStyle.NUMPYDOC)
        if not text:
            return ret
        text = inspect.cleandoc(text)
        match = self.titles_re.search(text)
        if match:
            desc_chunk = text[:match.start()]
            meta_chunk = text[match.start():]
        else:
            desc_chunk = text
            meta_chunk = ''
        parts = desc_chunk.split('\n', 1)
        ret.short_description = parts[0] or None
        if len(parts) > 1:
            long_desc_chunk = parts[1] or ''
            ret.blank_after_short_description = long_desc_chunk.startswith('\n')
            ret.blank_after_long_description = long_desc_chunk.endswith('\n\n')
            ret.long_description = long_desc_chunk.strip() or None
        for match, nextmatch in _pairwise(self.titles_re.finditer(meta_chunk)):
            title = next((g for g in match.groups() if g is not None))
            factory = self.sections[title]
            start = match.end()
            end = nextmatch.start() if nextmatch is not None else None
            ret.meta.extend(factory.parse(meta_chunk[start:end]))
        return ret