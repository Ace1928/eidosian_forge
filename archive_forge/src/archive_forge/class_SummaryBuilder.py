from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
class SummaryBuilder(object):
    """Class that builds a summary of certain attributes of a command.

  This will summarize a json representation of a command using
  cloud SDK-style markdown (but with no text wrapping) by taking snippets
  of the given locations in a command.

  If a lookup is given from terms to where they appear, then the snippets will
  include the relevant terms. Occurrences of search terms will be stylized.

  Uses a small amount of simple Cloud SDK markdown.

  1) To get a summary with just the brief help:
  SummaryBuilder(command, {'alligator': 'capsule'}).GetSummary()

  [no heading]
  {excerpt of command['capsule'] with first appearance of 'alligator'}

  2) To get a summary with a section (can be first-level or inside 'sections',
  which is the same as detailed_help):
  SummaryBuilder(command, {'': 'sections.SECTION_NAME'}).GetSummary()

  # SECTION_NAME
  {excerpt of 'SECTION_NAME' section of detailed help. If it is a list
   it will be joined by ', '.}

  3) To get a summary with a specific positional arg:
  SummaryBuilder(command, {'crocodile': 'positionals.myarg.name'}).GetSummary()

  # POSITIONALS
  myarg::
  {excerpt of 'myarg' positional help containing 'crocodile'}

  4) To get a summary with specific flags, possibly including choices/defaults:
  SummaryBuilder.GetSummary(command,
                            {'a': 'flags.--my-flag.choices',
                             'b': 'flags.--my-other-flag.default'})

  # FLAGS
  myflag::
  {excerpt of help} Choices: {comma-separated list of flag choices}
  myotherflag::
  {excerpt of help} Default: {flag default}

  Attributes:
    command: dict, a json representation of a command.
    found_terms_map: dict, mapping of terms to the locations where they are
      found, equivalent to the return value of
      CommandSearchResults.FoundTermsMap(). This map is found under "results"
      in the command resource returned by help-search. Locations have segments
      separated by dots, such as sections.DESCRIPTION. If the first segment is
      "flags" or "positionals", there must be three segments.
    length_per_snippet: int, length of desired substrings to get from text.
  """
    _INVALID_LOCATION_MESSAGE = 'Attempted to look up a location [{}] that was not found or invalid.'
    _IMPRECISE_LOCATION_MESSAGE = 'Expected location with three segments, received [{}]'

    def __init__(self, command, found_terms_map, length_per_snippet=200):
        """Create the class."""
        self.command = command
        self.found_terms_map = found_terms_map
        self.length_per_snippet = length_per_snippet
        self._lines = []

    def _AddFlagToSummary(self, location, terms):
        """Adds flag summary, given location such as ['flags']['--myflag']."""
        flags = self.command.get(location[0], {})
        line = ''
        assert len(location) > 2, self._IMPRECISE_LOCATION_MESSAGE.format(DOT.join(location))
        flag = flags.get(location[1])
        assert flag and (not flag[lookup.IS_HIDDEN]), self._INVALID_LOCATION_MESSAGE.format(DOT.join(location))
        if _FormatHeader(lookup.FLAGS) not in self._lines:
            self._lines.append(_FormatHeader(lookup.FLAGS))
        if _FormatItem(location[1]) not in self._lines:
            self._lines.append(_FormatItem(location[1]))
            desc_line = flag.get(lookup.DESCRIPTION, '')
            desc_line = _Snip(desc_line, self.length_per_snippet, terms)
            assert desc_line, self._INVALID_LOCATION_MESSAGE.format(DOT.join(location))
            line = desc_line
        if location[2] == lookup.DEFAULT:
            default = flags.get(location[1]).get(lookup.DEFAULT)
            if default:
                if line not in self._lines:
                    self._lines.append(line)
                if isinstance(default, dict):
                    default = ', '.join([x for x in sorted(default.keys())])
                elif isinstance(default, list):
                    default = ', '.join([x for x in default])
                line = 'Default: {}.'.format(default)
        else:
            valid_subattributes = [lookup.NAME, lookup.DESCRIPTION, lookup.CHOICES]
            assert location[2] in valid_subattributes, self._INVALID_LOCATION_MESSAGE.format(DOT.join(location))
        if line:
            self._lines.append(line)

    def _AddPositionalToSummary(self, location, terms):
        """Adds summary of arg, given location such as ['positionals']['myarg']."""
        positionals = self.command.get(lookup.POSITIONALS)
        line = ''
        assert len(location) > 2, self._IMPRECISE_LOCATION_MESSAGE.format(DOT.join(location))
        positionals = [p for p in positionals if p[lookup.NAME] == location[1]]
        assert positionals, self._INVALID_LOCATION_MESSAGE.format(DOT.join(location))
        if _FormatHeader(lookup.POSITIONALS) not in self._lines:
            self._lines.append(_FormatHeader(lookup.POSITIONALS))
        self._lines.append(_FormatItem(location[1]))
        positional = positionals[0]
        line = positional.get(lookup.DESCRIPTION, '')
        line = _Snip(line, self.length_per_snippet, terms)
        if line:
            self._lines.append(line)

    def _AddGenericSectionToSummary(self, location, terms):
        """Helper function for adding sections in the form ['loc1','loc2',...]."""
        section = self.command
        for loc in location:
            section = section.get(loc, {})
            if isinstance(section, str):
                line = section
            elif isinstance(section, list):
                line = ', '.join(sorted(section))
            elif isinstance(section, dict):
                line = ', '.join(sorted(section.keys()))
            else:
                line = six.text_type(section)
        assert line, self._INVALID_LOCATION_MESSAGE.format(DOT.join(location))
        header = _FormatHeader(location[-1])
        if header:
            self._lines.append(header)
        loc = '.'.join(location)
        self._lines.append(_Snip(line, self.length_per_snippet, terms))

    def GetSummary(self):
        """Builds a summary.

    Returns:
      str, a markdown summary
    """
        all_locations = set(self.found_terms_map.values())
        if lookup.CAPSULE not in all_locations:
            all_locations.add(lookup.CAPSULE)

        def _Equivalent(location, other_location):
            """Returns True if both locations correspond to same summary section."""
            if location == other_location:
                return True
            if len(location) != len(other_location):
                return False
            if location[:-1] != other_location[:-1]:
                return False
            equivalent = [lookup.NAME, lookup.CHOICES, lookup.DESCRIPTION]
            if location[-1] in equivalent and other_location[-1] in equivalent:
                return True
            return False
        for full_location in sorted(sorted(all_locations), key=_SummaryPriority):
            location = full_location.split(DOT)
            terms = {t for t, l in six.iteritems(self.found_terms_map) if _Equivalent(l.split(DOT), location) and t}
            if location[0] == lookup.FLAGS:
                self._AddFlagToSummary(location, terms)
            elif location[0] == lookup.POSITIONALS:
                self._AddPositionalToSummary(location, terms)
            elif lookup.PATH in location or lookup.NAME in location:
                continue
            else:
                self._AddGenericSectionToSummary(location, terms)
        summary = '\n'.join(self._lines)
        return Highlight(summary, self.found_terms_map.keys())