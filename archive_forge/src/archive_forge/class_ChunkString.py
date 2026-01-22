import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
class ChunkString:
    """
    A string-based encoding of a particular chunking of a text.
    Internally, the ``ChunkString`` class uses a single string to
    encode the chunking of the input text.  This string contains a
    sequence of angle-bracket delimited tags, with chunking indicated
    by braces.  An example of this encoding is::

        {<DT><JJ><NN>}<VBN><IN>{<DT><NN>}<.>{<DT><NN>}<VBD><.>

    ``ChunkString`` are created from tagged texts (i.e., lists of
    ``tokens`` whose type is ``TaggedType``).  Initially, nothing is
    chunked.

    The chunking of a ``ChunkString`` can be modified with the ``xform()``
    method, which uses a regular expression to transform the string
    representation.  These transformations should only add and remove
    braces; they should *not* modify the sequence of angle-bracket
    delimited tags.

    :type _str: str
    :ivar _str: The internal string representation of the text's
        encoding.  This string representation contains a sequence of
        angle-bracket delimited tags, with chunking indicated by
        braces.  An example of this encoding is::

            {<DT><JJ><NN>}<VBN><IN>{<DT><NN>}<.>{<DT><NN>}<VBD><.>

    :type _pieces: list(tagged tokens and chunks)
    :ivar _pieces: The tagged tokens and chunks encoded by this ``ChunkString``.
    :ivar _debug: The debug level.  See the constructor docs.

    :cvar IN_CHUNK_PATTERN: A zero-width regexp pattern string that
        will only match positions that are in chunks.
    :cvar IN_STRIP_PATTERN: A zero-width regexp pattern string that
        will only match positions that are in strips.
    """
    CHUNK_TAG_CHAR = '[^\\{\\}<>]'
    CHUNK_TAG = '(<%s+?>)' % CHUNK_TAG_CHAR
    IN_CHUNK_PATTERN = '(?=[^\\{]*\\})'
    IN_STRIP_PATTERN = '(?=[^\\}]*(\\{|$))'
    _CHUNK = '(\\{%s+?\\})+?' % CHUNK_TAG
    _STRIP = '(%s+?)+?' % CHUNK_TAG
    _VALID = re.compile('^(\\{?%s\\}?)*?$' % CHUNK_TAG)
    _BRACKETS = re.compile('[^\\{\\}]+')
    _BALANCED_BRACKETS = re.compile('(\\{\\})*$')

    def __init__(self, chunk_struct, debug_level=1):
        """
        Construct a new ``ChunkString`` that encodes the chunking of
        the text ``tagged_tokens``.

        :type chunk_struct: Tree
        :param chunk_struct: The chunk structure to be further chunked.
        :type debug_level: int
        :param debug_level: The level of debugging which should be
            applied to transformations on the ``ChunkString``.  The
            valid levels are:

                - 0: no checks
                - 1: full check on to_chunkstruct
                - 2: full check on to_chunkstruct and cursory check after
                  each transformation.
                - 3: full check on to_chunkstruct and full check after
                  each transformation.

            We recommend you use at least level 1.  You should
            probably use level 3 if you use any non-standard
            subclasses of ``RegexpChunkRule``.
        """
        self._root_label = chunk_struct.label()
        self._pieces = chunk_struct[:]
        tags = [self._tag(tok) for tok in self._pieces]
        self._str = '<' + '><'.join(tags) + '>'
        self._debug = debug_level

    def _tag(self, tok):
        if isinstance(tok, tuple):
            return tok[1]
        elif isinstance(tok, Tree):
            return tok.label()
        else:
            raise ValueError('chunk structures must contain tagged tokens or trees')

    def _verify(self, s, verify_tags):
        """
        Check to make sure that ``s`` still corresponds to some chunked
        version of ``_pieces``.

        :type verify_tags: bool
        :param verify_tags: Whether the individual tags should be
            checked.  If this is false, ``_verify`` will check to make
            sure that ``_str`` encodes a chunked version of *some*
            list of tokens.  If this is true, then ``_verify`` will
            check to make sure that the tags in ``_str`` match those in
            ``_pieces``.

        :raise ValueError: if the internal string representation of
            this ``ChunkString`` is invalid or not consistent with _pieces.
        """
        if not ChunkString._VALID.match(s):
            raise ValueError('Transformation generated invalid chunkstring:\n  %s' % s)
        brackets = ChunkString._BRACKETS.sub('', s)
        for i in range(1 + len(brackets) // 5000):
            substr = brackets[i * 5000:i * 5000 + 5000]
            if not ChunkString._BALANCED_BRACKETS.match(substr):
                raise ValueError('Transformation generated invalid chunkstring:\n  %s' % s)
        if verify_tags <= 0:
            return
        tags1 = re.split('[\\{\\}<>]+', s)[1:-1]
        tags2 = [self._tag(piece) for piece in self._pieces]
        if tags1 != tags2:
            raise ValueError('Transformation generated invalid chunkstring: tag changed')

    def to_chunkstruct(self, chunk_label='CHUNK'):
        """
        Return the chunk structure encoded by this ``ChunkString``.

        :rtype: Tree
        :raise ValueError: If a transformation has generated an
            invalid chunkstring.
        """
        if self._debug > 0:
            self._verify(self._str, 1)
        pieces = []
        index = 0
        piece_in_chunk = 0
        for piece in re.split('[{}]', self._str):
            length = piece.count('<')
            subsequence = self._pieces[index:index + length]
            if piece_in_chunk:
                pieces.append(Tree(chunk_label, subsequence))
            else:
                pieces += subsequence
            index += length
            piece_in_chunk = not piece_in_chunk
        return Tree(self._root_label, pieces)

    def xform(self, regexp, repl):
        """
        Apply the given transformation to the string encoding of this
        ``ChunkString``.  In particular, find all occurrences that match
        ``regexp``, and replace them using ``repl`` (as done by
        ``re.sub``).

        This transformation should only add and remove braces; it
        should *not* modify the sequence of angle-bracket delimited
        tags.  Furthermore, this transformation may not result in
        improper bracketing.  Note, in particular, that bracketing may
        not be nested.

        :type regexp: str or regexp
        :param regexp: A regular expression matching the substring
            that should be replaced.  This will typically include a
            named group, which can be used by ``repl``.
        :type repl: str
        :param repl: An expression specifying what should replace the
            matched substring.  Typically, this will include a named
            replacement group, specified by ``regexp``.
        :rtype: None
        :raise ValueError: If this transformation generated an
            invalid chunkstring.
        """
        s = re.sub(regexp, repl, self._str)
        s = re.sub('\\{\\}', '', s)
        if self._debug > 1:
            self._verify(s, self._debug - 2)
        self._str = s

    def __repr__(self):
        """
        Return a string representation of this ``ChunkString``.
        It has the form::

            <ChunkString: '{<DT><JJ><NN>}<VBN><IN>{<DT><NN>}'>

        :rtype: str
        """
        return '<ChunkString: %s>' % repr(self._str)

    def __str__(self):
        """
        Return a formatted representation of this ``ChunkString``.
        This representation will include extra spaces to ensure that
        tags will line up with the representation of other
        ``ChunkStrings`` for the same text, regardless of the chunking.

        :rtype: str
        """
        str = re.sub('>(?!\\})', '> ', self._str)
        str = re.sub('([^\\{])<', '\\1 <', str)
        if str[0] == '<':
            str = ' ' + str
        return str