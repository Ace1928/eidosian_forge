import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
class ReviewsCorpusReader(CorpusReader):
    """
    Reader for the Customer Review Data dataset by Hu, Liu (2004).
    Note: we are not applying any sentence tokenization at the moment, just word
    tokenization.

        >>> from nltk.corpus import product_reviews_1
        >>> camera_reviews = product_reviews_1.reviews('Canon_G3.txt')
        >>> review = camera_reviews[0]
        >>> review.sents()[0] # doctest: +NORMALIZE_WHITESPACE
        ['i', 'recently', 'purchased', 'the', 'canon', 'powershot', 'g3', 'and', 'am',
        'extremely', 'satisfied', 'with', 'the', 'purchase', '.']
        >>> review.features() # doctest: +NORMALIZE_WHITESPACE
        [('canon powershot g3', '+3'), ('use', '+2'), ('picture', '+2'),
        ('picture quality', '+1'), ('picture quality', '+1'), ('camera', '+2'),
        ('use', '+2'), ('feature', '+1'), ('picture quality', '+3'), ('use', '+1'),
        ('option', '+1')]

    We can also reach the same information directly from the stream:

        >>> product_reviews_1.features('Canon_G3.txt')
        [('canon powershot g3', '+3'), ('use', '+2'), ...]

    We can compute stats for specific product features:

        >>> n_reviews = len([(feat,score) for (feat,score) in product_reviews_1.features('Canon_G3.txt') if feat=='picture'])
        >>> tot = sum([int(score) for (feat,score) in product_reviews_1.features('Canon_G3.txt') if feat=='picture'])
        >>> mean = tot / n_reviews
        >>> print(n_reviews, tot, mean)
        15 24 1.6
    """
    CorpusView = StreamBackedCorpusView

    def __init__(self, root, fileids, word_tokenizer=WordPunctTokenizer(), encoding='utf8'):
        """
        :param root: The root directory for the corpus.
        :param fileids: a list or regexp specifying the fileids in the corpus.
        :param word_tokenizer: a tokenizer for breaking sentences or paragraphs
            into words. Default: `WordPunctTokenizer`
        :param encoding: the encoding that should be used to read the corpus.
        """
        CorpusReader.__init__(self, root, fileids, encoding)
        self._word_tokenizer = word_tokenizer
        self._readme = 'README.txt'

    def features(self, fileids=None):
        """
        Return a list of features. Each feature is a tuple made of the specific
        item feature and the opinion strength about that feature.

        :param fileids: a list or regexp specifying the ids of the files whose
            features have to be returned.
        :return: all features for the item(s) in the given file(s).
        :rtype: list(tuple)
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat([self.CorpusView(fileid, self._read_features, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def reviews(self, fileids=None):
        """
        Return all the reviews as a list of Review objects. If `fileids` is
        specified, return all the reviews from each of the specified files.

        :param fileids: a list or regexp specifying the ids of the files whose
            reviews have to be returned.
        :return: the given file(s) as a list of reviews.
        """
        if fileids is None:
            fileids = self._fileids
        return concat([self.CorpusView(fileid, self._read_review_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def sents(self, fileids=None):
        """
        Return all sentences in the corpus or in the specified files.

        :param fileids: a list or regexp specifying the ids of the files whose
            sentences have to be returned.
        :return: the given file(s) as a list of sentences, each encoded as a
            list of word strings.
        :rtype: list(list(str))
        """
        return concat([self.CorpusView(path, self._read_sent_block, encoding=enc) for path, enc, fileid in self.abspaths(fileids, True, True)])

    def words(self, fileids=None):
        """
        Return all words and punctuation symbols in the corpus or in the specified
        files.

        :param fileids: a list or regexp specifying the ids of the files whose
            words have to be returned.
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        """
        return concat([self.CorpusView(path, self._read_word_block, encoding=enc) for path, enc, fileid in self.abspaths(fileids, True, True)])

    def _read_features(self, stream):
        features = []
        for i in range(20):
            line = stream.readline()
            if not line:
                return features
            features.extend(re.findall(FEATURES, line))
        return features

    def _read_review_block(self, stream):
        while True:
            line = stream.readline()
            if not line:
                return []
            title_match = re.match(TITLE, line)
            if title_match:
                review = Review(title=title_match.group(1).strip())
                break
        while True:
            oldpos = stream.tell()
            line = stream.readline()
            if not line:
                return [review]
            if re.match(TITLE, line):
                stream.seek(oldpos)
                return [review]
            feats = re.findall(FEATURES, line)
            notes = re.findall(NOTES, line)
            sent = re.findall(SENT, line)
            if sent:
                sent = self._word_tokenizer.tokenize(sent[0])
            review_line = ReviewLine(sent=sent, features=feats, notes=notes)
            review.add_line(review_line)

    def _read_sent_block(self, stream):
        sents = []
        for review in self._read_review_block(stream):
            sents.extend([sent for sent in review.sents()])
        return sents

    def _read_word_block(self, stream):
        words = []
        for i in range(20):
            line = stream.readline()
            sent = re.findall(SENT, line)
            if sent:
                words.extend(self._word_tokenizer.tokenize(sent[0]))
        return words