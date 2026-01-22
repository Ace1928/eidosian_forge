import subprocess
from collections import namedtuple
class AlignedSent:
    """
    Return an aligned sentence object, which encapsulates two sentences
    along with an ``Alignment`` between them.

    Typically used in machine translation to represent a sentence and
    its translation.

        >>> from nltk.translate import AlignedSent, Alignment
        >>> algnsent = AlignedSent(['klein', 'ist', 'das', 'Haus'],
        ...     ['the', 'house', 'is', 'small'], Alignment.fromstring('0-3 1-2 2-0 3-1'))
        >>> algnsent.words
        ['klein', 'ist', 'das', 'Haus']
        >>> algnsent.mots
        ['the', 'house', 'is', 'small']
        >>> algnsent.alignment
        Alignment([(0, 3), (1, 2), (2, 0), (3, 1)])
        >>> from nltk.corpus import comtrans
        >>> print(comtrans.aligned_sents()[54])
        <AlignedSent: 'Weshalb also sollten...' -> 'So why should EU arm...'>
        >>> print(comtrans.aligned_sents()[54].alignment)
        0-0 0-1 1-0 2-2 3-4 3-5 4-7 5-8 6-3 7-9 8-9 9-10 9-11 10-12 11-6 12-6 13-13

    :param words: Words in the target language sentence
    :type words: list(str)
    :param mots: Words in the source language sentence
    :type mots: list(str)
    :param alignment: Word-level alignments between ``words`` and ``mots``.
        Each alignment is represented as a 2-tuple (words_index, mots_index).
    :type alignment: Alignment
    """

    def __init__(self, words, mots, alignment=None):
        self._words = words
        self._mots = mots
        if alignment is None:
            self.alignment = Alignment([])
        else:
            assert type(alignment) is Alignment
            self.alignment = alignment

    @property
    def words(self):
        return self._words

    @property
    def mots(self):
        return self._mots

    def _get_alignment(self):
        return self._alignment

    def _set_alignment(self, alignment):
        _check_alignment(len(self.words), len(self.mots), alignment)
        self._alignment = alignment
    alignment = property(_get_alignment, _set_alignment)

    def __repr__(self):
        """
        Return a string representation for this ``AlignedSent``.

        :rtype: str
        """
        words = '[%s]' % ', '.join(("'%s'" % w for w in self._words))
        mots = '[%s]' % ', '.join(("'%s'" % w for w in self._mots))
        return f'AlignedSent({words}, {mots}, {self._alignment!r})'

    def _to_dot(self):
        """
        Dot representation of the aligned sentence
        """
        s = 'graph align {\n'
        s += 'node[shape=plaintext]\n'
        for w in self._words:
            s += f'"{w}_source" [label="{w}"] \n'
        for w in self._mots:
            s += f'"{w}_target" [label="{w}"] \n'
        for u, v in self._alignment:
            s += f'"{self._words[u]}_source" -- "{self._mots[v]}_target" \n'
        for i in range(len(self._words) - 1):
            s += '"{}_source" -- "{}_source" [style=invis]\n'.format(self._words[i], self._words[i + 1])
        for i in range(len(self._mots) - 1):
            s += '"{}_target" -- "{}_target" [style=invis]\n'.format(self._mots[i], self._mots[i + 1])
        s += '{rank = same; %s}\n' % ' '.join(('"%s_source"' % w for w in self._words))
        s += '{rank = same; %s}\n' % ' '.join(('"%s_target"' % w for w in self._mots))
        s += '}'
        return s

    def _repr_svg_(self):
        """
        Ipython magic : show SVG representation of this ``AlignedSent``.
        """
        dot_string = self._to_dot().encode('utf8')
        output_format = 'svg'
        try:
            process = subprocess.Popen(['dot', '-T%s' % output_format], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            raise Exception('Cannot find the dot binary from Graphviz package') from e
        out, err = process.communicate(dot_string)
        return out.decode('utf8')

    def __str__(self):
        """
        Return a human-readable string representation for this ``AlignedSent``.

        :rtype: str
        """
        source = ' '.join(self._words)[:20] + '...'
        target = ' '.join(self._mots)[:20] + '...'
        return f"<AlignedSent: '{source}' -> '{target}'>"

    def invert(self):
        """
        Return the aligned sentence pair, reversing the directionality

        :rtype: AlignedSent
        """
        return AlignedSent(self._mots, self._words, self._alignment.invert())