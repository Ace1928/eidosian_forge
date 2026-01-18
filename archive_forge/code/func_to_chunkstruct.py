import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
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