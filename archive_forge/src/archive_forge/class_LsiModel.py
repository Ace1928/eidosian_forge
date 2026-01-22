import logging
import sys
import time
import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse import sparsetools
from gensim import interfaces, matutils, utils
from gensim.models import basemodel
from gensim.utils import is_empty
class LsiModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """Model for `Latent Semantic Indexing
    <https://en.wikipedia.org/wiki/Latent_semantic_analysis#Latent_semantic_indexing>`_.

    The decomposition algorithm is described in `"Fast and Faster: A Comparison of Two Streamed
    Matrix Decomposition Algorithms" <https://arxiv.org/pdf/1102.5597.pdf>`_.

    Notes
    -----
    * :attr:`gensim.models.lsimodel.LsiModel.projection.u` - left singular vectors,
    * :attr:`gensim.models.lsimodel.LsiModel.projection.s` - singular values,
    * ``model[training_corpus]`` - right singular vectors (can be reconstructed if needed).

    See Also
    --------
    `FAQ about LSI matrices
    <https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ#q4-how-do-you-output-the-u-s-vt-matrices-of-lsi>`_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
        >>> from gensim.models import LsiModel
        >>>
        >>> model = LsiModel(common_corpus[:3], id2word=common_dictionary)  # train model
        >>> vector = model[common_corpus[4]]  # apply model to BoW document
        >>> model.add_documents(common_corpus[4:])  # update model with new documents
        >>> tmp_fname = get_tmpfile("lsi.model")
        >>> model.save(tmp_fname)  # save model
        >>> loaded_model = LsiModel.load(tmp_fname)  # load model

    """

    def __init__(self, corpus=None, num_topics=200, id2word=None, chunksize=20000, decay=1.0, distributed=False, onepass=True, power_iters=P2_EXTRA_ITERS, extra_samples=P2_EXTRA_DIMS, dtype=np.float64, random_seed=None):
        """Build an LSI model.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or a sparse matrix of shape (`num_documents`, `num_terms`).
        num_topics : int, optional
            Number of requested factors (latent dimensions)
        id2word : dict of {int: str}, optional
            ID to word mapping, optional.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        decay : float, optional
            Weight of existing observations relatively to new ones.
        distributed : bool, optional
            If True - distributed mode (parallel execution on several machines) will be used.
        onepass : bool, optional
            Whether the one-pass algorithm should be used for training.
            Pass `False` to force a multi-pass stochastic algorithm.
        power_iters: int, optional
            Number of power iteration steps to be used.
            Increasing the number of power iterations improves accuracy, but lowers performance
        extra_samples : int, optional
            Extra samples to be used besides the rank `k`. Can improve accuracy.
        dtype : type, optional
            Enforces a type for elements of the decomposed matrix.
        random_seed: {None, int}, optional
            Random seed used to initialize the pseudo-random number generator,
            a local instance of numpy.random.RandomState instance.

        """
        self.id2word = id2word
        self.num_topics = int(num_topics)
        self.chunksize = int(chunksize)
        self.decay = float(decay)
        if distributed:
            if not onepass:
                logger.warning('forcing the one-pass algorithm for distributed LSA')
                onepass = True
        self.onepass = onepass
        self.extra_samples, self.power_iters = (extra_samples, power_iters)
        self.dtype = dtype
        self.random_seed = random_seed
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')
        if self.id2word is None:
            logger.warning('no word id mapping provided; initializing from corpus, assuming identity')
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 1 + (max(self.id2word.keys()) if self.id2word else -1)
        self.docs_processed = 0
        self.projection = Projection(self.num_terms, self.num_topics, power_iters=self.power_iters, extra_dims=self.extra_samples, dtype=dtype, random_seed=self.random_seed)
        self.numworkers = 1
        if not distributed:
            logger.info('using serial LSI version on this node')
            self.dispatcher = None
        else:
            if not onepass:
                raise NotImplementedError('distributed stochastic LSA not implemented yet; run either distributed one-pass, or serial randomized.')
            try:
                import Pyro4
                dispatcher = Pyro4.Proxy('PYRONAME:gensim.lsi_dispatcher')
                logger.debug('looking for dispatcher at %s', str(dispatcher._pyroUri))
                dispatcher.initialize(id2word=self.id2word, num_topics=num_topics, chunksize=chunksize, decay=decay, power_iters=self.power_iters, extra_samples=self.extra_samples, distributed=False, onepass=onepass)
                self.dispatcher = dispatcher
                self.numworkers = len(dispatcher.getworkers())
                logger.info('using distributed version with %i workers', self.numworkers)
            except Exception as err:
                logger.error('failed to initialize distributed LSI (%s)', err)
                raise RuntimeError('failed to initialize distributed LSI (%s)' % err)
        if corpus is not None:
            start = time.time()
            self.add_documents(corpus)
            self.add_lifecycle_event('created', msg=f'trained {self} in {time.time() - start:.2f}s')

    def add_documents(self, corpus, chunksize=None, decay=None):
        """Update model with new `corpus`.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}
            Stream of document vectors or sparse matrix of shape (`num_terms`, num_documents).
        chunksize : int, optional
            Number of documents to be used in each training chunk, will use `self.chunksize` if not specified.
        decay : float, optional
            Weight of existing observations relatively to new ones,  will use `self.decay` if not specified.

        Notes
        -----
        Training proceeds in chunks of `chunksize` documents at a time. The size of `chunksize` is a tradeoff
        between increased speed (bigger `chunksize`) vs. lower memory footprint (smaller `chunksize`).
        If the distributed mode is on, each chunk is sent to a different worker/computer.

        """
        logger.info('updating model with new documents')
        if chunksize is None:
            chunksize = self.chunksize
        if decay is None:
            decay = self.decay
        if is_empty(corpus):
            logger.warning('LsiModel.add_documents() called but no documents provided, is this intended?')
        if not scipy.sparse.issparse(corpus):
            if not self.onepass:
                update = Projection(self.num_terms, self.num_topics, None, dtype=self.dtype, random_seed=self.random_seed)
                update.u, update.s = stochastic_svd(corpus, self.num_topics, num_terms=self.num_terms, chunksize=chunksize, extra_dims=self.extra_samples, power_iters=self.power_iters, dtype=self.dtype, random_seed=self.random_seed)
                self.projection.merge(update, decay=decay)
                self.docs_processed += len(corpus) if hasattr(corpus, '__len__') else 0
            else:
                doc_no = 0
                if self.dispatcher:
                    logger.info('initializing %s workers', self.numworkers)
                    self.dispatcher.reset()
                for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize)):
                    logger.info('preparing a new chunk of documents')
                    nnz = sum((len(doc) for doc in chunk))
                    logger.debug('converting corpus to csc format')
                    job = matutils.corpus2csc(chunk, num_docs=len(chunk), num_terms=self.num_terms, num_nnz=nnz, dtype=self.dtype)
                    del chunk
                    doc_no += job.shape[1]
                    if self.dispatcher:
                        logger.debug('creating job #%i', chunk_no)
                        self.dispatcher.putjob(job)
                        del job
                        logger.info('dispatched documents up to #%s', doc_no)
                    else:
                        update = Projection(self.num_terms, self.num_topics, job, extra_dims=self.extra_samples, power_iters=self.power_iters, dtype=self.dtype, random_seed=self.random_seed)
                        del job
                        self.projection.merge(update, decay=decay)
                        del update
                        logger.info('processed documents up to #%s', doc_no)
                        self.print_topics(5)
                if self.dispatcher:
                    logger.info('reached the end of input; now waiting for all remaining jobs to finish')
                    self.projection = self.dispatcher.getstate()
                self.docs_processed += doc_no
        else:
            assert not self.dispatcher, 'must be in serial mode to receive jobs'
            update = Projection(self.num_terms, self.num_topics, corpus.tocsc(), extra_dims=self.extra_samples, power_iters=self.power_iters, dtype=self.dtype)
            self.projection.merge(update, decay=decay)
            logger.info('processed sparse job of %i documents', corpus.shape[1])
            self.docs_processed += corpus.shape[1]

    def __str__(self):
        """Get a human readable representation of model.

        Returns
        -------
        str
            A human readable string of the current objects parameters.

        """
        return '%s<num_terms=%s, num_topics=%s, decay=%s, chunksize=%s>' % (self.__class__.__name__, self.num_terms, self.num_topics, self.decay, self.chunksize)

    def __getitem__(self, bow, scaled=False, chunksize=512):
        """Get the latent representation for `bow`.

        Parameters
        ----------
        bow : {list of (int, int), iterable of list of (int, int)}
            Document or corpus in BoW representation.
        scaled : bool, optional
            If True - topics will be scaled by the inverse of singular values.
        chunksize :  int, optional
            Number of documents to be used in each applying chunk.

        Returns
        -------
        list of (int, float)
            Latent representation of topics in BoW format for document **OR**
        :class:`gensim.matutils.Dense2Corpus`
            Latent representation of corpus in BoW format if `bow` is corpus.

        """
        if self.projection.u is None:
            raise ValueError('No training data provided - LSI model not initialized yet')
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus and chunksize:
            return self._apply(bow, chunksize=chunksize)
        if not is_corpus:
            bow = [bow]
        vec = matutils.corpus2csc(bow, num_terms=self.num_terms, dtype=self.projection.u.dtype)
        topic_dist = (vec.T * self.projection.u[:, :self.num_topics]).T
        if not is_corpus:
            topic_dist = topic_dist.reshape(-1)
        if scaled:
            topic_dist = 1.0 / self.projection.s[:self.num_topics] * topic_dist
        if not is_corpus:
            result = matutils.full2sparse(topic_dist)
        else:
            result = matutils.Dense2Corpus(topic_dist)
        return result

    def get_topics(self):
        """Get the topic vectors.

        Notes
        -----
        The number of topics can actually be smaller than `self.num_topics`, if there were not enough factors
        in the matrix (real rank of input matrix smaller than `self.num_topics`).

        Returns
        -------
        np.ndarray
            The term topic matrix with shape (`num_topics`, `vocabulary_size`)

        """
        projections = self.projection.u.T
        num_topics = len(projections)
        topics = []
        for i in range(num_topics):
            c = np.asarray(projections[i, :]).flatten()
            norm = np.sqrt(np.sum(np.dot(c, c)))
            topics.append(1.0 * c / norm)
        return np.array(topics)

    def show_topic(self, topicno, topn=10):
        """Get the words that define a topic along with their contribution.

        This is actually the left singular vector of the specified topic.

        The most important words in defining the topic (greatest absolute value) are included
        in the output, along with their contribution to the topic.

        Parameters
        ----------
        topicno : int
            The topics id number.
        topn : int
            Number of words to be included to the result.

        Returns
        -------
        list of (str, float)
            Topic representation in BoW format.

        """
        if topicno >= len(self.projection.u.T):
            return ''
        c = np.asarray(self.projection.u.T[topicno, :]).flatten()
        norm = np.sqrt(np.sum(np.dot(c, c)))
        most = matutils.argsort(np.abs(c), topn, reverse=True)
        return [(self.id2word[val], 1.0 * c[val] / norm) for val in most if val in self.id2word]

    def show_topics(self, num_topics=-1, num_words=10, log=False, formatted=True):
        """Get the most significant topics.

        Parameters
        ----------
        num_topics : int, optional
            The number of topics to be selected, if -1 - all topics will be in result (ordered by significance).
        num_words : int, optional
            The number of words to be included per topics (ordered by significance).
        log : bool, optional
            If True - log topics with logger.
        formatted : bool, optional
            If True - each topic represented as string, otherwise - in BoW format.

        Returns
        -------
        list of (int, str)
            If `formatted=True`, return sequence with (topic_id, string representation of topics) **OR**
        list of (int, list of (str, float))
            Otherwise, return sequence with (topic_id, [(word, value), ... ]).

        """
        shown = []
        if num_topics < 0:
            num_topics = self.num_topics
        for i in range(min(num_topics, self.num_topics)):
            if i < len(self.projection.s):
                if formatted:
                    topic = self.print_topic(i, topn=num_words)
                else:
                    topic = self.show_topic(i, topn=num_words)
                shown.append((i, topic))
                if log:
                    logger.info('topic #%i(%.3f): %s', i, self.projection.s[i], topic)
        return shown

    def print_debug(self, num_topics=5, num_words=10):
        """Print (to log) the most salient words of the first `num_topics` topics.

        Unlike :meth:`~gensim.models.lsimodel.LsiModel.print_topics`, this looks for words that are significant for
        a particular topic *and* not for others. This *should* result in a
        more human-interpretable description of topics.

        Alias for :func:`~gensim.models.lsimodel.print_debug`.

        Parameters
        ----------
        num_topics : int, optional
            The number of topics to be selected (ordered by significance).
        num_words : int, optional
            The number of words to be included per topics (ordered by significance).

        """
        print_debug(self.id2word, self.projection.u, self.projection.s, range(min(num_topics, len(self.projection.u.T))), num_words=num_words)

    def save(self, fname, *args, **kwargs):
        """Save the model to a file.

        Notes
        -----
        Large internal arrays may be stored into separate files, with `fname` as prefix.

        Warnings
        --------
        Do not save as a compressed file if you intend to load the file back with `mmap`.

        Parameters
        ----------
        fname : str
            Path to output file.
        *args
            Variable length argument list, see :meth:`gensim.utils.SaveLoad.save`.
        **kwargs
            Arbitrary keyword arguments, see :meth:`gensim.utils.SaveLoad.save`.

        See Also
        --------
        :meth:`~gensim.models.lsimodel.LsiModel.load`

        """
        if self.projection is not None:
            self.projection.save(utils.smart_extension(fname, '.projection'), *args, **kwargs)
        super(LsiModel, self).save(fname, *args, ignore=['projection', 'dispatcher'], **kwargs)

    @classmethod
    def load(cls, fname, *args, **kwargs):
        """Load a previously saved object using :meth:`~gensim.models.lsimodel.LsiModel.save` from file.

        Notes
        -----
        Large arrays can be memmap'ed back as read-only (shared memory) by setting the `mmap='r'` parameter.

        Parameters
        ----------
        fname : str
            Path to file that contains LsiModel.
        *args
            Variable length argument list, see :meth:`gensim.utils.SaveLoad.load`.
        **kwargs
            Arbitrary keyword arguments, see :meth:`gensim.utils.SaveLoad.load`.

        See Also
        --------
        :meth:`~gensim.models.lsimodel.LsiModel.save`

        Returns
        -------
        :class:`~gensim.models.lsimodel.LsiModel`
            Loaded instance.

        Raises
        ------
        IOError
            When methods are called on instance (should be called from class).

        """
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(LsiModel, cls).load(fname, *args, **kwargs)
        projection_fname = utils.smart_extension(fname, '.projection')
        try:
            result.projection = super(LsiModel, cls).load(projection_fname, *args, **kwargs)
        except Exception as e:
            logging.warning('failed to load projection from %s: %s', projection_fname, e)
        return result