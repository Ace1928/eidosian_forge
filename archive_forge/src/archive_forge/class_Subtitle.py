from __future__ import unicode_literals
import functools
import re
from datetime import timedelta
import logging
import io
@functools.total_ordering
class Subtitle(object):
    """
    The metadata relating to a single subtitle. Subtitles are sorted by start
    time by default. If no index was provided, index 0 will be used on writing
    an SRT block.

    :param index: The SRT index for this subtitle
    :type index: int or None
    :param start: The time that the subtitle should start being shown
    :type start: :py:class:`datetime.timedelta`
    :param end: The time that the subtitle should stop being shown
    :type end: :py:class:`datetime.timedelta`
    :param str proprietary: Proprietary metadata for this subtitle
    :param str content: The subtitle content. Should not contain OS-specific
                        line separators, only \\\\n. This is taken care of
                        already if you use :py:func:`srt.parse` to generate
                        Subtitle objects.
    """

    def __init__(self, index, start, end, content, proprietary=''):
        self.index = index
        self.start = start
        self.end = end
        self.content = content
        self.proprietary = proprietary

    def __hash__(self):
        return hash(frozenset(vars(self).items()))

    def __eq__(self, other):
        return vars(self) == vars(other)

    def __lt__(self, other):
        return (self.start, self.end, self.index) < (other.start, other.end, other.index)

    def __repr__(self):
        var_items = getattr(vars(self), 'iteritems', getattr(vars(self), 'items'))
        item_list = ', '.join(('%s=%r' % (k, v) for k, v in var_items()))
        return '%s(%s)' % (type(self).__name__, item_list)

    def to_srt(self, strict=True, eol='\n'):
        """
        Convert the current :py:class:`Subtitle` to an SRT block.

        :param bool strict: If disabled, will allow blank lines in the content
                            of the SRT block, which is a violation of the SRT
                            standard and may cause your media player to explode
        :param str eol: The end of line string to use (default "\\\\n")
        :returns: The metadata of the current :py:class:`Subtitle` object as an
                  SRT formatted subtitle block
        :rtype: str
        """
        output_content = self.content
        output_proprietary = self.proprietary
        if output_proprietary:
            output_proprietary = ' ' + output_proprietary
        if strict:
            output_content = make_legal_content(output_content)
        if eol is None:
            eol = '\n'
        elif eol != '\n':
            output_content = output_content.replace('\n', eol)
        template = '{idx}{eol}{start} --> {end}{prop}{eol}{content}{eol}{eol}'
        return template.format(idx=self.index or 0, start=timedelta_to_srt_timestamp(self.start), end=timedelta_to_srt_timestamp(self.end), prop=output_proprietary, content=output_content, eol=eol)