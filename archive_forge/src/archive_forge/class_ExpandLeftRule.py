import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
class ExpandLeftRule(RegexpChunkRule):
    """
    A rule specifying how to expand chunks in a ``ChunkString`` to the left,
    using two matching tag patterns: a left pattern, and a right pattern.
    When applied to a ``ChunkString``, it will find any chunk whose beginning
    matches right pattern, and immediately preceded by a strip whose
    end matches left pattern.  It will then expand the chunk to incorporate
    the new material on the left.
    """

    def __init__(self, left_tag_pattern, right_tag_pattern, descr):
        """
        Construct a new ``ExpandRightRule``.

        :type right_tag_pattern: str
        :param right_tag_pattern: This rule's right tag
            pattern.  When applied to a ``ChunkString``, this
            rule will find any chunk whose beginning matches
            ``right_tag_pattern``, and immediately preceded by a strip
            whose end matches this pattern.  It will
            then merge those two chunks into a single chunk.
        :type left_tag_pattern: str
        :param left_tag_pattern: This rule's left tag
            pattern.  When applied to a ``ChunkString``, this
            rule will find any chunk whose beginning matches
            this pattern, and immediately preceded by a strip
            whose end matches ``left_tag_pattern``.  It will
            then expand the chunk to incorporate the new material on the left.

        :type descr: str
        :param descr: A short description of the purpose and/or effect
            of this rule.
        """
        re.compile(tag_pattern2re_pattern(left_tag_pattern))
        re.compile(tag_pattern2re_pattern(right_tag_pattern))
        self._left_tag_pattern = left_tag_pattern
        self._right_tag_pattern = right_tag_pattern
        regexp = re.compile('(?P<left>%s)\\{(?P<right>%s)' % (tag_pattern2re_pattern(left_tag_pattern), tag_pattern2re_pattern(right_tag_pattern)))
        RegexpChunkRule.__init__(self, regexp, '{\\g<left>\\g<right>', descr)

    def __repr__(self):
        """
        Return a string representation of this rule.  It has the form::

            <ExpandLeftRule: '<NN|DT|JJ>', '<NN|JJ>'>

        Note that this representation does not include the
        description string; that string can be accessed
        separately with the ``descr()`` method.

        :rtype: str
        """
        return '<ExpandLeftRule: ' + repr(self._left_tag_pattern) + ', ' + repr(self._right_tag_pattern) + '>'