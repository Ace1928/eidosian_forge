import logging
from collections import Counter
from gensim import utils
from gensim.corpora import IndexedCorpus
from gensim.parsing.preprocessing import split_on_space
class LowCorpus(IndexedCorpus):
    """Corpus handles input in `GibbsLda++ format <https://gibbslda.sourceforge.net/>`_.

    **Format description**

    Both data for training/estimating the model and new data (i.e., previously unseen data) have the same format
    as follows ::

        [M]
        [document1]
        [document2]
        ...
        [documentM]

    in which the first line is the total number for documents [M]. Each line after that is one document.
    [documenti] is the ith document of the dataset that consists of a list of Ni words/terms ::

        [documenti] = [wordi1] [wordi2] ... [wordiNi]

    in which all [wordij] (i=1..M, j=1..Ni) are text strings and they are separated by the blank character.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import get_tmpfile, common_texts
        >>> from gensim.corpora import LowCorpus
        >>> from gensim.corpora import Dictionary
        >>>
        >>> # Prepare needed data
        >>> dictionary = Dictionary(common_texts)
        >>> corpus = [dictionary.doc2bow(doc) for doc in common_texts]
        >>>
        >>> # Write corpus in GibbsLda++ format to disk
        >>> output_fname = get_tmpfile("corpus.low")
        >>> LowCorpus.serialize(output_fname, corpus, dictionary)
        >>>
        >>> # Read corpus
        >>> loaded_corpus = LowCorpus(output_fname)

    """

    def __init__(self, fname, id2word=None, line2words=split_on_space):
        """

        Parameters
        ----------
        fname : str
            Path to file in GibbsLda++ format.
        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional
            Mapping between word_ids (integers) and words (strings).
            If not provided, the mapping is constructed directly from `fname`.
        line2words : callable, optional
            Function which converts lines(str) into tokens(list of str),
            using :func:`~gensim.parsing.preprocessing.split_on_space` as default.

        """
        IndexedCorpus.__init__(self, fname)
        logger.info('loading corpus from %s', fname)
        self.fname = fname
        self.line2words = line2words
        self.num_docs = self._calculate_num_docs()
        if not id2word:
            logger.info('extracting vocabulary from the corpus')
            all_terms = set()
            self.use_wordids = False
            for doc in self:
                all_terms.update((word for word, wordCnt in doc))
            all_terms = sorted(all_terms)
            self.id2word = dict(zip(range(len(all_terms)), all_terms))
        else:
            logger.info('using provided word mapping (%i ids)', len(id2word))
            self.id2word = id2word
        self.num_terms = len(self.word2id)
        self.use_wordids = True
        logger.info('loaded corpus with %i documents and %i terms from %s', self.num_docs, self.num_terms, fname)

    def _calculate_num_docs(self):
        """Get number of documents in file.

        Returns
        -------
        int
            Number of documents.

        """
        with utils.open(self.fname, 'rb') as fin:
            try:
                result = int(next(fin))
            except StopIteration:
                result = 0
        return result

    def __len__(self):
        return self.num_docs

    def line2doc(self, line):
        """Covert line into document in BoW format.

        Parameters
        ----------
        line : str
            Line from input file.

        Returns
        -------
        list of (int, int)
            Document in BoW format

        """
        words = self.line2words(line)
        if self.use_wordids:
            use_words, counts = ([], Counter())
            for word in words:
                if word not in self.word2id:
                    continue
                if word not in counts:
                    use_words.append(word)
                counts[word] += 1
            doc = [(self.word2id[w], counts[w]) for w in use_words]
        else:
            word_freqs = Counter(words)
            doc = list(word_freqs.items())
        return doc

    def __iter__(self):
        """Iterate over the corpus.

        Yields
        ------
        list of (int, int)
            Document in BoW format.

        """
        with utils.open(self.fname, 'rb') as fin:
            for lineno, line in enumerate(fin):
                if lineno > 0:
                    yield self.line2doc(line)

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        """Save a corpus in the GibbsLda++ format.

        Warnings
        --------
        This function is automatically called by :meth:`gensim.corpora.lowcorpus.LowCorpus.serialize`,
        don't call it directly, call :meth:`gensim.corpora.lowcorpus.LowCorpus.serialize` instead.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format.
        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional
            Mapping between word_ids (integers) and words (strings).
            If not provided, the mapping is constructed directly from `corpus`.
        metadata : bool, optional
            THIS PARAMETER WILL BE IGNORED.

        Return
        ------
        list of int
            List of offsets in resulting file for each document (in bytes),
            can be used for :meth:`~gensim.corpora.lowcorpus.LowCorpus.docbyoffset`

        """
        if id2word is None:
            logger.info('no word id mapping provided; initializing from corpus')
            id2word = utils.dict_from_corpus(corpus)
        logger.info('storing corpus in List-Of-Words format into %s' % fname)
        truncated = 0
        offsets = []
        with utils.open(fname, 'wb') as fout:
            fout.write(utils.to_utf8('%i\n' % len(corpus)))
            for doc in corpus:
                words = []
                for wordid, value in doc:
                    if abs(int(value) - value) > 1e-06:
                        truncated += 1
                    words.extend([utils.to_unicode(id2word[wordid])] * int(value))
                offsets.append(fout.tell())
                fout.write(utils.to_utf8('%s\n' % ' '.join(words)))
        if truncated:
            logger.warning('List-of-words format can only save vectors with integer elements; %i float entries were truncated to integer value', truncated)
        return offsets

    def docbyoffset(self, offset):
        """Get the document stored in file by `offset` position.

        Parameters
        ----------
        offset : int
            Offset (in bytes) to begin of document.

        Returns
        -------
        list of (int, int)
            Document in BoW format.

        Examples
        --------

        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.corpora import LowCorpus
            >>>
            >>> data = LowCorpus(datapath("testcorpus.low"))
            >>> data.docbyoffset(1)  # end of first line
            []
            >>> data.docbyoffset(2)  # start of second line
            [(0, 1), (3, 1), (4, 1)]

        """
        with utils.open(self.fname, 'rb') as f:
            f.seek(offset)
            return self.line2doc(f.readline())

    @property
    def id2word(self):
        """Get mapping between words and their ids."""
        return self._id2word

    @id2word.setter
    def id2word(self, val):
        self._id2word = val
        self.word2id = utils.revdict(val)