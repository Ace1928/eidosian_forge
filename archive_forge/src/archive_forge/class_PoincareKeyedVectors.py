import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
class PoincareKeyedVectors(KeyedVectors):
    """Vectors and vocab for the :class:`~gensim.models.poincare.PoincareModel` training class.

    Used to perform operations on the vectors such as vector lookup, distance calculations etc.

    (May be used to save/load final vectors in the plain word2vec format, via the inherited
    methods save_word2vec_format() and load_word2vec_format().)

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath
        >>>
        >>> # Read the sample relations file and train the model
        >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
        >>> model = PoincareModel(train_data=relations)
        >>> model.train(epochs=50)
        >>>
        >>> # Query the trained model.
        >>> wv = model.kv.get_vector('kangaroo.n.01')

    """

    def __init__(self, vector_size, vector_count, dtype=REAL):
        super(PoincareKeyedVectors, self).__init__(vector_size, vector_count, dtype=dtype)
        self.max_distance = 0

    def _load_specials(self, *args, **kwargs):
        super(PoincareKeyedVectors, self)._load_specials(*args, **kwargs)
        if not hasattr(self, 'vectors'):
            self.vectors = self.__dict__.pop('syn0')

    @staticmethod
    def vector_distance(vector_1, vector_2):
        """Compute poincare distance between two input vectors. Convenience method over `vector_distance_batch`.

        Parameters
        ----------
        vector_1 : numpy.array
            Input vector.
        vector_2 : numpy.array
            Input vector.

        Returns
        -------
        numpy.float
            Poincare distance between `vector_1` and `vector_2`.

        """
        return PoincareKeyedVectors.vector_distance_batch(vector_1, vector_2[np.newaxis, :])[0]

    @staticmethod
    def vector_distance_batch(vector_1, vectors_all):
        """Compute poincare distances between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.array
            vector from which Poincare distances are to be computed, expected shape (dim,).
        vectors_all : numpy.array
            for each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).

        Returns
        -------
        numpy.array
            Poincare distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).

        """
        euclidean_dists = np.linalg.norm(vector_1 - vectors_all, axis=1)
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        return np.arccosh(1 + 2 * (euclidean_dists ** 2 / ((1 - norm ** 2) * (1 - all_norms ** 2))))

    def closest_child(self, node):
        """Get the node closest to `node` that is lower in the hierarchy than `node`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which closest child is to be found.

        Returns
        -------
        {str, None}
            Node closest to `node` that is lower in the hierarchy than `node`.
            If there are no nodes lower in the hierarchy, None is returned.

        """
        all_distances = self.distances(node)
        all_norms = np.linalg.norm(self.vectors, axis=1)
        node_norm = all_norms[self.get_index(node)]
        mask = node_norm >= all_norms
        if mask.all():
            return None
        all_distances = np.ma.array(all_distances, mask=mask)
        closest_child_index = np.ma.argmin(all_distances)
        return self.index_to_key[closest_child_index]

    def closest_parent(self, node):
        """Get the node closest to `node` that is higher in the hierarchy than `node`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which closest parent is to be found.

        Returns
        -------
        {str, None}
            Node closest to `node` that is higher in the hierarchy than `node`.
            If there are no nodes higher in the hierarchy, None is returned.

        """
        all_distances = self.distances(node)
        all_norms = np.linalg.norm(self.vectors, axis=1)
        node_norm = all_norms[self.get_index(node)]
        mask = node_norm <= all_norms
        if mask.all():
            return None
        all_distances = np.ma.array(all_distances, mask=mask)
        closest_child_index = np.ma.argmin(all_distances)
        return self.index_to_key[closest_child_index]

    def descendants(self, node, max_depth=5):
        """Get the list of recursively closest children from the given node, up to a max depth of `max_depth`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which descendants are to be found.
        max_depth : int
            Maximum number of descendants to return.

        Returns
        -------
        list of str
            Descendant nodes from the node `node`.

        """
        depth = 0
        descendants = []
        current_node = node
        while depth < max_depth:
            descendants.append(self.closest_child(current_node))
            current_node = descendants[-1]
            depth += 1
        return descendants

    def ancestors(self, node):
        """Get the list of recursively closest parents from the given node.

        Parameters
        ----------
        node : {str, int}
            Key for node for which ancestors are to be found.

        Returns
        -------
        list of str
            Ancestor nodes of the node `node`.

        """
        ancestors = []
        current_node = node
        ancestor = self.closest_parent(current_node)
        while ancestor is not None:
            ancestors.append(ancestor)
            ancestor = self.closest_parent(ancestors[-1])
        return ancestors

    def distance(self, w1, w2):
        """Calculate Poincare distance between vectors for nodes `w1` and `w2`.

        Parameters
        ----------
        w1 : {str, int}
            Key for first node.
        w2 : {str, int}
            Key for second node.

        Returns
        -------
        float
            Poincare distance between the vectors for nodes `w1` and `w2`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # What is the distance between the words 'mammal' and 'carnivore'?
            >>> model.kv.distance('mammal.n.01', 'carnivore.n.01')
            2.9742298803339304

        Raises
        ------
        KeyError
            If either of `w1` and `w2` is absent from vocab.

        """
        vector_1 = self.get_vector(w1)
        vector_2 = self.get_vector(w2)
        return self.vector_distance(vector_1, vector_2)

    def similarity(self, w1, w2):
        """Compute similarity based on Poincare distance between vectors for nodes `w1` and `w2`.

        Parameters
        ----------
        w1 : {str, int}
            Key for first node.
        w2 : {str, int}
            Key for second node.

        Returns
        -------
        float
            Similarity between the between the vectors for nodes `w1` and `w2` (between 0 and 1).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # What is the similarity between the words 'mammal' and 'carnivore'?
            >>> model.kv.similarity('mammal.n.01', 'carnivore.n.01')
            0.25162107631176484

        Raises
        ------
        KeyError
            If either of `w1` and `w2` is absent from vocab.

        """
        return 1 / (1 + self.distance(w1, w2))

    def most_similar(self, node_or_vector, topn=10, restrict_vocab=None):
        """Find the top-N most similar nodes to the given node or vector, sorted in increasing order of distance.

        Parameters
        ----------
        node_or_vector : {str, int, numpy.array}
            node key or vector for which similar nodes are to be found.
        topn : int or None, optional
            Number of top-N similar nodes to return, when `topn` is int. When `topn` is None,
            then distance for all nodes are returned.
        restrict_vocab : int or None, optional
            Optional integer which limits the range of vectors which are searched for most-similar values.
            For example, restrict_vocab=10000 would only check the first 10000 node vectors in the vocabulary order.
            This may be meaningful if vocabulary is sorted by descending frequency.

        Returns
        --------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (node, distance) is returned in increasing order of distance.
            When `topn` is None, then similarities for all words are returned as a one-dimensional numpy array with the
            size of the vocabulary.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Which words are most similar to 'kangaroo'?
            >>> model.kv.most_similar('kangaroo.n.01', topn=2)
            [(u'kangaroo.n.01', 0.0), (u'marsupial.n.01', 0.26524229460827725)]

        """
        if isinstance(topn, Integral) and topn < 1:
            return []
        if not restrict_vocab:
            all_distances = self.distances(node_or_vector)
        else:
            nodes_to_use = self.index_to_key[:restrict_vocab]
            all_distances = self.distances(node_or_vector, nodes_to_use)
        if isinstance(node_or_vector, (str, int)):
            node_index = self.get_index(node_or_vector)
        else:
            node_index = None
        if not topn:
            closest_indices = matutils.argsort(all_distances)
        else:
            closest_indices = matutils.argsort(all_distances, topn=1 + topn)
        result = [(self.index_to_key[index], float(all_distances[index])) for index in closest_indices if not node_index or index != node_index]
        if topn:
            result = result[:topn]
        return result

    def distances(self, node_or_vector, other_nodes=()):
        """Compute Poincare distances from given `node_or_vector` to all nodes in `other_nodes`.
        If `other_nodes` is empty, return distance between `node_or_vector` and all nodes in vocab.

        Parameters
        ----------
        node_or_vector : {str, int, numpy.array}
            Node key or vector from which distances are to be computed.
        other_nodes : {iterable of str, iterable of int, None}, optional
            For each node in `other_nodes` distance from `node_or_vector` is computed.
            If None or empty, distance of `node_or_vector` from all nodes in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all nodes in `other_nodes` from input `node_or_vector`,
            in the same order as `other_nodes`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Check the distances between a word and a list of other words.
            >>> model.kv.distances('mammal.n.01', ['carnivore.n.01', 'dog.n.01'])
            array([2.97422988, 2.83007402])

            >>> # Check the distances between a word and every other word in the vocab.
            >>> all_distances = model.kv.distances('mammal.n.01')

        Raises
        ------
        KeyError
            If either `node_or_vector` or any node in `other_nodes` is absent from vocab.

        """
        if isinstance(node_or_vector, str):
            input_vector = self.get_vector(node_or_vector)
        else:
            input_vector = node_or_vector
        if not other_nodes:
            other_vectors = self.vectors
        else:
            other_indices = [self.get_index(node) for node in other_nodes]
            other_vectors = self.vectors[other_indices]
        return self.vector_distance_batch(input_vector, other_vectors)

    def norm(self, node_or_vector):
        """Compute absolute position in hierarchy of input node or vector.
        Values range between 0 and 1. A lower value indicates the input node or vector is higher in the hierarchy.

        Parameters
        ----------
        node_or_vector : {str, int, numpy.array}
            Input node key or vector for which position in hierarchy is to be returned.

        Returns
        -------
        float
            Absolute position in the hierarchy of the input vector or node.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Get the norm of the embedding of the word `mammal`.
            >>> model.kv.norm('mammal.n.01')
            0.6423008703542398

        Notes
        -----
        The position in hierarchy is based on the norm of the vector for the node.

        """
        if isinstance(node_or_vector, str):
            input_vector = self.get_vector(node_or_vector)
        else:
            input_vector = node_or_vector
        return np.linalg.norm(input_vector)

    def difference_in_hierarchy(self, node_or_vector_1, node_or_vector_2):
        """Compute relative position in hierarchy of `node_or_vector_1` relative to `node_or_vector_2`.
        A positive value indicates `node_or_vector_1` is higher in the hierarchy than `node_or_vector_2`.

        Parameters
        ----------
        node_or_vector_1 : {str, int, numpy.array}
            Input node key or vector.
        node_or_vector_2 : {str, int, numpy.array}
            Input node key or vector.

        Returns
        -------
        float
            Relative position in hierarchy of `node_or_vector_1` relative to `node_or_vector_2`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> model.kv.difference_in_hierarchy('mammal.n.01', 'dog.n.01')
            0.05382517902410999

            >>> model.kv.difference_in_hierarchy('dog.n.01', 'mammal.n.01')
            -0.05382517902410999

        Notes
        -----
        The returned value can be positive or negative, depending on whether `node_or_vector_1` is higher
        or lower in the hierarchy than `node_or_vector_2`.

        """
        return self.norm(node_or_vector_2) - self.norm(node_or_vector_1)