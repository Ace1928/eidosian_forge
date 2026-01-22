import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
class RegexpChunkRule:
    """
    A rule specifying how to modify the chunking in a ``ChunkString``,
    using a transformational regular expression.  The
    ``RegexpChunkRule`` class itself can be used to implement any
    transformational rule based on regular expressions.  There are
    also a number of subclasses, which can be used to implement
    simpler types of rules, based on matching regular expressions.

    Each ``RegexpChunkRule`` has a regular expression and a
    replacement expression.  When a ``RegexpChunkRule`` is "applied"
    to a ``ChunkString``, it searches the ``ChunkString`` for any
    substring that matches the regular expression, and replaces it
    using the replacement expression.  This search/replace operation
    has the same semantics as ``re.sub``.

    Each ``RegexpChunkRule`` also has a description string, which
    gives a short (typically less than 75 characters) description of
    the purpose of the rule.

    This transformation defined by this ``RegexpChunkRule`` should
    only add and remove braces; it should *not* modify the sequence
    of angle-bracket delimited tags.  Furthermore, this transformation
    may not result in nested or mismatched bracketing.
    """

    def __init__(self, regexp, repl, descr):
        """
        Construct a new RegexpChunkRule.

        :type regexp: regexp or str
        :param regexp: The regular expression for this ``RegexpChunkRule``.
            When this rule is applied to a ``ChunkString``, any
            substring that matches ``regexp`` will be replaced using
            the replacement string ``repl``.  Note that this must be a
            normal regular expression, not a tag pattern.
        :type repl: str
        :param repl: The replacement expression for this ``RegexpChunkRule``.
            When this rule is applied to a ``ChunkString``, any substring
            that matches ``regexp`` will be replaced using ``repl``.
        :type descr: str
        :param descr: A short description of the purpose and/or effect
            of this rule.
        """
        if isinstance(regexp, str):
            regexp = re.compile(regexp)
        self._repl = repl
        self._descr = descr
        self._regexp = regexp

    def apply(self, chunkstr):
        """
        Apply this rule to the given ``ChunkString``.  See the
        class reference documentation for a description of what it
        means to apply a rule.

        :type chunkstr: ChunkString
        :param chunkstr: The chunkstring to which this rule is applied.
        :rtype: None
        :raise ValueError: If this transformation generated an
            invalid chunkstring.
        """
        chunkstr.xform(self._regexp, self._repl)

    def descr(self):
        """
        Return a short description of the purpose and/or effect of
        this rule.

        :rtype: str
        """
        return self._descr

    def __repr__(self):
        """
        Return a string representation of this rule.  It has the form::

            <RegexpChunkRule: '{<IN|VB.*>}'->'<IN>'>

        Note that this representation does not include the
        description string; that string can be accessed
        separately with the ``descr()`` method.

        :rtype: str
        """
        return '<RegexpChunkRule: ' + repr(self._regexp.pattern) + '->' + repr(self._repl) + '>'

    @staticmethod
    def fromstring(s):
        """
        Create a RegexpChunkRule from a string description.
        Currently, the following formats are supported::

          {regexp}         # chunk rule
          }regexp{         # strip rule
          regexp}{regexp   # split rule
          regexp{}regexp   # merge rule

        Where ``regexp`` is a regular expression for the rule.  Any
        text following the comment marker (``#``) will be used as
        the rule's description:

        >>> from nltk.chunk.regexp import RegexpChunkRule
        >>> RegexpChunkRule.fromstring('{<DT>?<NN.*>+}')
        <ChunkRule: '<DT>?<NN.*>+'>
        """
        m = re.match('(?P<rule>(\\\\.|[^#])*)(?P<comment>#.*)?', s)
        rule = m.group('rule').strip()
        comment = (m.group('comment') or '')[1:].strip()
        try:
            if not rule:
                raise ValueError('Empty chunk pattern')
            if rule[0] == '{' and rule[-1] == '}':
                return ChunkRule(rule[1:-1], comment)
            elif rule[0] == '}' and rule[-1] == '{':
                return StripRule(rule[1:-1], comment)
            elif '}{' in rule:
                left, right = rule.split('}{')
                return SplitRule(left, right, comment)
            elif '{}' in rule:
                left, right = rule.split('{}')
                return MergeRule(left, right, comment)
            elif re.match('[^{}]*{[^{}]*}[^{}]*', rule):
                left, chunk, right = re.split('[{}]', rule)
                return ChunkRuleWithContext(left, chunk, right, comment)
            else:
                raise ValueError('Illegal chunk pattern: %s' % rule)
        except (ValueError, re.error) as e:
            raise ValueError('Illegal chunk pattern: %s' % rule) from e