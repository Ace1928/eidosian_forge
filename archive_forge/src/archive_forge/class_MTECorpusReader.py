import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
class MTECorpusReader(TaggedCorpusReader):
    """
    Reader for corpora following the TEI-p5 xml scheme, such as MULTEXT-East.
    MULTEXT-East contains part-of-speech-tagged words with a quite precise tagging
    scheme. These tags can be converted to the Universal tagset
    """

    def __init__(self, root=None, fileids=None, encoding='utf8'):
        """
        Construct a new MTECorpusreader for a set of documents
        located at the given root directory.  Example usage:

            >>> root = '/...path to corpus.../'
            >>> reader = MTECorpusReader(root, 'oana-*.xml', 'utf8') # doctest: +SKIP

        :param root: The root directory for this corpus. (default points to location in multext config file)
        :param fileids: A list or regexp specifying the fileids in this corpus. (default is oana-en.xml)
        :param encoding: The encoding of the given files (default is utf8)
        """
        TaggedCorpusReader.__init__(self, root, fileids, encoding)
        self._readme = '00README.txt'

    def __fileids(self, fileids):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        fileids = filter(lambda x: x in self._fileids, fileids)
        fileids = filter(lambda x: x not in ['oana-bg.xml', 'oana-mk.xml'], fileids)
        if not fileids:
            print('No valid multext-east file specified')
        return fileids

    def words(self, fileids=None):
        """
        :param fileids: A list specifying the fileids that should be used.
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        """
        return concat([MTEFileReader(os.path.join(self._root, f)).words() for f in self.__fileids(fileids)])

    def sents(self, fileids=None):
        """
        :param fileids: A list specifying the fileids that should be used.
        :return: the given file(s) as a list of sentences or utterances,
                 each encoded as a list of word strings
        :rtype: list(list(str))
        """
        return concat([MTEFileReader(os.path.join(self._root, f)).sents() for f in self.__fileids(fileids)])

    def paras(self, fileids=None):
        """
        :param fileids: A list specifying the fileids that should be used.
        :return: the given file(s) as a list of paragraphs, each encoded as a list
                 of sentences, which are in turn encoded as lists of word string
        :rtype: list(list(list(str)))
        """
        return concat([MTEFileReader(os.path.join(self._root, f)).paras() for f in self.__fileids(fileids)])

    def lemma_words(self, fileids=None):
        """
        :param fileids: A list specifying the fileids that should be used.
        :return: the given file(s) as a list of words, the corresponding lemmas
                 and punctuation symbols, encoded as tuples (word, lemma)
        :rtype: list(tuple(str,str))
        """
        return concat([MTEFileReader(os.path.join(self._root, f)).lemma_words() for f in self.__fileids(fileids)])

    def tagged_words(self, fileids=None, tagset='msd', tags=''):
        """
        :param fileids: A list specifying the fileids that should be used.
        :param tagset: The tagset that should be used in the returned object,
                       either "universal" or "msd", "msd" is the default
        :param tags: An MSD Tag that is used to filter all parts of the used corpus
                     that are not more precise or at least equal to the given tag
        :return: the given file(s) as a list of tagged words and punctuation symbols
                 encoded as tuples (word, tag)
        :rtype: list(tuple(str, str))
        """
        if tagset == 'universal' or tagset == 'msd':
            return concat([MTEFileReader(os.path.join(self._root, f)).tagged_words(tagset, tags) for f in self.__fileids(fileids)])
        else:
            print('Unknown tagset specified.')

    def lemma_sents(self, fileids=None):
        """
        :param fileids: A list specifying the fileids that should be used.
        :return: the given file(s) as a list of sentences or utterances, each
                 encoded as a list of tuples of the word and the corresponding
                 lemma (word, lemma)
        :rtype: list(list(tuple(str, str)))
        """
        return concat([MTEFileReader(os.path.join(self._root, f)).lemma_sents() for f in self.__fileids(fileids)])

    def tagged_sents(self, fileids=None, tagset='msd', tags=''):
        """
        :param fileids: A list specifying the fileids that should be used.
        :param tagset: The tagset that should be used in the returned object,
                       either "universal" or "msd", "msd" is the default
        :param tags: An MSD Tag that is used to filter all parts of the used corpus
                     that are not more precise or at least equal to the given tag
        :return: the given file(s) as a list of sentences or utterances, each
                 each encoded as a list of (word,tag) tuples
        :rtype: list(list(tuple(str, str)))
        """
        if tagset == 'universal' or tagset == 'msd':
            return concat([MTEFileReader(os.path.join(self._root, f)).tagged_sents(tagset, tags) for f in self.__fileids(fileids)])
        else:
            print('Unknown tagset specified.')

    def lemma_paras(self, fileids=None):
        """
        :param fileids: A list specifying the fileids that should be used.
        :return: the given file(s) as a list of paragraphs, each encoded as a
                 list of sentences, which are in turn encoded as a list of
                 tuples of the word and the corresponding lemma (word, lemma)
        :rtype: list(List(List(tuple(str, str))))
        """
        return concat([MTEFileReader(os.path.join(self._root, f)).lemma_paras() for f in self.__fileids(fileids)])

    def tagged_paras(self, fileids=None, tagset='msd', tags=''):
        """
        :param fileids: A list specifying the fileids that should be used.
        :param tagset: The tagset that should be used in the returned object,
                       either "universal" or "msd", "msd" is the default
        :param tags: An MSD Tag that is used to filter all parts of the used corpus
                     that are not more precise or at least equal to the given tag
        :return: the given file(s) as a list of paragraphs, each encoded as a
                 list of sentences, which are in turn encoded as a list
                 of (word,tag) tuples
        :rtype: list(list(list(tuple(str, str))))
        """
        if tagset == 'universal' or tagset == 'msd':
            return concat([MTEFileReader(os.path.join(self._root, f)).tagged_paras(tagset, tags) for f in self.__fileids(fileids)])
        else:
            print('Unknown tagset specified.')