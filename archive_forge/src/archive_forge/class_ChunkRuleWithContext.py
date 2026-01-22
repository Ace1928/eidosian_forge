import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
class ChunkRuleWithContext(RegexpChunkRule):
    """
    A rule specifying how to add chunks to a ``ChunkString``, using
    three matching tag patterns: one for the left context, one for the
    chunk, and one for the right context.  When applied to a
    ``ChunkString``, it will find any substring that matches the chunk
    tag pattern, is surrounded by substrings that match the two
    context patterns, and is not already part of a chunk; and create a
    new chunk containing the substring that matched the chunk tag
    pattern.

    Caveat: Both the left and right context are consumed when this
    rule matches; therefore, if you need to find overlapping matches,
    you will need to apply your rule more than once.
    """

    def __init__(self, left_context_tag_pattern, chunk_tag_pattern, right_context_tag_pattern, descr):
        """
        Construct a new ``ChunkRuleWithContext``.

        :type left_context_tag_pattern: str
        :param left_context_tag_pattern: A tag pattern that must match
            the left context of ``chunk_tag_pattern`` for this rule to
            apply.
        :type chunk_tag_pattern: str
        :param chunk_tag_pattern: A tag pattern that must match for this
            rule to apply.  If the rule does apply, then this pattern
            also identifies the substring that will be made into a chunk.
        :type right_context_tag_pattern: str
        :param right_context_tag_pattern: A tag pattern that must match
            the right context of ``chunk_tag_pattern`` for this rule to
            apply.
        :type descr: str
        :param descr: A short description of the purpose and/or effect
            of this rule.
        """
        re.compile(tag_pattern2re_pattern(left_context_tag_pattern))
        re.compile(tag_pattern2re_pattern(chunk_tag_pattern))
        re.compile(tag_pattern2re_pattern(right_context_tag_pattern))
        self._left_context_tag_pattern = left_context_tag_pattern
        self._chunk_tag_pattern = chunk_tag_pattern
        self._right_context_tag_pattern = right_context_tag_pattern
        regexp = re.compile('(?P<left>%s)(?P<chunk>%s)(?P<right>%s)%s' % (tag_pattern2re_pattern(left_context_tag_pattern), tag_pattern2re_pattern(chunk_tag_pattern), tag_pattern2re_pattern(right_context_tag_pattern), ChunkString.IN_STRIP_PATTERN))
        replacement = '\\g<left>{\\g<chunk>}\\g<right>'
        RegexpChunkRule.__init__(self, regexp, replacement, descr)

    def __repr__(self):
        """
        Return a string representation of this rule.  It has the form::

            <ChunkRuleWithContext: '<IN>', '<NN>', '<DT>'>

        Note that this representation does not include the
        description string; that string can be accessed
        separately with the ``descr()`` method.

        :rtype: str
        """
        return '<ChunkRuleWithContext:  {!r}, {!r}, {!r}>'.format(self._left_context_tag_pattern, self._chunk_tag_pattern, self._right_context_tag_pattern)