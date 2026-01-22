import argparse
import collections
import contextlib
import io
import re
import tokenize
from typing import TextIO, Tuple
import untokenize  # type: ignore
import docformatter.encode as _encode
import docformatter.strings as _strings
import docformatter.syntax as _syntax
import docformatter.util as _util
def _do_format_docstring(self, indentation: str, docstring: str) -> str:
    """Return formatted version of docstring.

        Parameters
        ----------
        indentation: str
            The indentation characters for the docstring.
        docstring: str
            The docstring itself.

        Returns
        -------
        docstring_formatted: str
            The docstring formatted according the various options.
        """
    contents, open_quote = self._do_strip_docstring(docstring)
    if self.args.black and contents.startswith('"') or (not self.args.black and self.args.pre_summary_space):
        open_quote = f'{open_quote} '
    if contents.count(self.QUOTE_TYPES[0]):
        return docstring
    if contents.lstrip().startswith('>>>'):
        return docstring
    _links = _syntax.do_find_links(contents)
    with contextlib.suppress(IndexError):
        if _links[0][0] == 0 and _links[0][1] == len(contents):
            return docstring
    summary, description = _strings.split_summary_and_description(contents)
    if _syntax.is_some_sort_of_field_list(summary, self.args.style):
        return docstring
    if _syntax.remove_section_header(description).strip() != description.strip():
        return docstring
    if not self.args.force_wrap and (_syntax.is_some_sort_of_list(summary, self.args.non_strict, self.args.rest_section_adorns, self.args.style) or _syntax.do_find_links(summary)):
        return docstring
    tab_compensation = indentation.count('\t') * (self.args.tab_width - 1)
    self.args.wrap_summaries -= tab_compensation
    self.args.wrap_descriptions -= tab_compensation
    if description:
        return self._do_format_multiline_docstring(indentation, summary, description, open_quote)
    return self._do_format_oneline_docstring(indentation, contents, open_quote)