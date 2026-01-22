import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
class UnChunkRule(RegexpChunkRule):
    """
    A rule specifying how to remove chunks to a ``ChunkString``,
    using a matching tag pattern.  When applied to a
    ``ChunkString``, it will find any complete chunk that matches this
    tag pattern, and un-chunk it.
    """

    def __init__(self, tag_pattern, descr):
        """
        Construct a new ``UnChunkRule``.

        :type tag_pattern: str
        :param tag_pattern: This rule's tag pattern.  When
            applied to a ``ChunkString``, this rule will
            find any complete chunk that matches this tag pattern,
            and un-chunk it.
        :type descr: str
        :param descr: A short description of the purpose and/or effect
            of this rule.
        """
        self._pattern = tag_pattern
        regexp = re.compile('\\{(?P<chunk>%s)\\}' % tag_pattern2re_pattern(tag_pattern))
        RegexpChunkRule.__init__(self, regexp, '\\g<chunk>', descr)

    def __repr__(self):
        """
        Return a string representation of this rule.  It has the form::

            <UnChunkRule: '<IN|VB.*>'>

        Note that this representation does not include the
        description string; that string can be accessed
        separately with the ``descr()`` method.

        :rtype: str
        """
        return '<UnChunkRule: ' + repr(self._pattern) + '>'