import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
class LdaPost(utils.SaveLoad):
    """Posterior values associated with each set of documents.

    TODO: use **Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.**
    to update phi, gamma. End game would be to somehow replace LdaPost entirely with LdaModel.

    """

    def __init__(self, doc=None, lda=None, max_doc_len=None, num_topics=None, gamma=None, lhood=None):
        """Initialize the posterior value structure for the given LDA model.

        Parameters
        ----------
        doc : list of (int, int)
            A BOW representation of the document. Each element in the list is a pair of a word's ID and its number
            of occurences in the document.
        lda : :class:`~gensim.models.ldamodel.LdaModel`, optional
            The underlying LDA model.
        max_doc_len : int, optional
            The maximum number of words in a document.
        num_topics : int, optional
            Number of topics discovered by the LDA model.
        gamma : numpy.ndarray, optional
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        lhood : float, optional
            The log likelihood lower bound.

        """
        self.doc = doc
        self.lda = lda
        self.gamma = gamma
        self.lhood = lhood
        if self.gamma is None:
            self.gamma = np.zeros(num_topics)
        if self.lhood is None:
            self.lhood = np.zeros(num_topics + 1)
        if max_doc_len is not None and num_topics is not None:
            self.phi = np.zeros((max_doc_len, num_topics))
            self.log_phi = np.zeros((max_doc_len, num_topics))
        self.doc_weight = None
        self.renormalized_doc_weight = None

    def update_phi(self, doc_number, time):
        """Update variational multinomial parameters, based on a document and a time-slice.

        This is done based on the original Blei-LDA paper, where:
        log_phi := beta * exp(Ψ(gamma)), over every topic for every word.

        TODO: incorporate lee-sueng trick used in
        **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.

        Parameters
        ----------
        doc_number : int
            Document number. Unused.
        time : int
            Time slice. Unused.

        Returns
        -------
        (list of float, list of float)
            Multinomial parameters, and their logarithm, for each word in the document.

        """
        num_topics = self.lda.num_topics
        dig = np.zeros(num_topics)
        for k in range(num_topics):
            dig[k] = digamma(self.gamma[k])
        n = 0
        for word_id, count in self.doc:
            for k in range(num_topics):
                self.log_phi[n][k] = dig[k] + self.lda.topics[word_id][k]
            log_phi_row = self.log_phi[n]
            phi_row = self.phi[n]
            v = log_phi_row[0]
            for i in range(1, len(log_phi_row)):
                v = np.logaddexp(v, log_phi_row[i])
            log_phi_row = log_phi_row - v
            phi_row = np.exp(log_phi_row)
            self.log_phi[n] = log_phi_row
            self.phi[n] = phi_row
            n += 1
        return (self.phi, self.log_phi)

    def update_gamma(self):
        """Update variational dirichlet parameters.

        This operations is described in the original Blei LDA paper:
        gamma = alpha + sum(phi), over every topic for every word.

        Returns
        -------
        list of float
            The updated gamma parameters for each word in the document.

        """
        self.gamma = np.copy(self.lda.alpha)
        n = 0
        for word_id, count in self.doc:
            phi_row = self.phi[n]
            for k in range(self.lda.num_topics):
                self.gamma[k] += phi_row[k] * count
            n += 1
        return self.gamma

    def init_lda_post(self):
        """Initialize variational posterior. """
        total = sum((count for word_id, count in self.doc))
        self.gamma.fill(self.lda.alpha[0] + float(total) / self.lda.num_topics)
        self.phi[:len(self.doc), :] = 1.0 / self.lda.num_topics

    def compute_lda_lhood(self):
        """Compute the log likelihood bound.

        Returns
        -------
        float
            The optimal lower bound for the true posterior using the approximate distribution.

        """
        num_topics = self.lda.num_topics
        gamma_sum = np.sum(self.gamma)
        lhood = gammaln(np.sum(self.lda.alpha)) - gammaln(gamma_sum)
        self.lhood[num_topics] = lhood
        digsum = digamma(gamma_sum)
        model = 'DTM'
        for k in range(num_topics):
            e_log_theta_k = digamma(self.gamma[k]) - digsum
            lhood_term = (self.lda.alpha[k] - self.gamma[k]) * e_log_theta_k + gammaln(self.gamma[k]) - gammaln(self.lda.alpha[k])
            n = 0
            for word_id, count in self.doc:
                if self.phi[n][k] > 0:
                    lhood_term += count * self.phi[n][k] * (e_log_theta_k + self.lda.topics[word_id][k] - self.log_phi[n][k])
                n += 1
            self.lhood[k] = lhood_term
            lhood += lhood_term
        return lhood

    def fit_lda_post(self, doc_number, time, ldaseq, LDA_INFERENCE_CONVERGED=1e-08, lda_inference_max_iter=25, g=None, g3_matrix=None, g4_matrix=None, g5_matrix=None):
        """Posterior inference for lda.

        Parameters
        ----------
        doc_number : int
            The documents number.
        time : int
            Time slice.
        ldaseq : object
            Unused.
        LDA_INFERENCE_CONVERGED : float
            Epsilon value used to check whether the inference step has sufficiently converged.
        lda_inference_max_iter : int
            Maximum number of iterations in the inference step.
        g : object
            Unused. Will be useful when the DIM model is implemented.
        g3_matrix: object
            Unused. Will be useful when the DIM model is implemented.
        g4_matrix: object
            Unused. Will be useful when the DIM model is implemented.
        g5_matrix: object
            Unused. Will be useful when the DIM model is implemented.

        Returns
        -------
        float
            The optimal lower bound for the true posterior using the approximate distribution.
        """
        self.init_lda_post()
        total = sum((count for word_id, count in self.doc))
        model = 'DTM'
        if model == 'DIM':
            pass
        lhood = self.compute_lda_lhood()
        lhood_old = 0
        converged = 0
        iter_ = 0
        iter_ += 1
        lhood_old = lhood
        self.gamma = self.update_gamma()
        model = 'DTM'
        if model == 'DTM' or sslm is None:
            self.phi, self.log_phi = self.update_phi(doc_number, time)
        elif model == 'DIM' and sslm is not None:
            self.phi, self.log_phi = self.update_phi_fixed(doc_number, time, sslm, g3_matrix, g4_matrix, g5_matrix)
        lhood = self.compute_lda_lhood()
        converged = np.fabs((lhood_old - lhood) / (lhood_old * total))
        while converged > LDA_INFERENCE_CONVERGED and iter_ <= lda_inference_max_iter:
            iter_ += 1
            lhood_old = lhood
            self.gamma = self.update_gamma()
            model = 'DTM'
            if model == 'DTM' or sslm is None:
                self.phi, self.log_phi = self.update_phi(doc_number, time)
            elif model == 'DIM' and sslm is not None:
                self.phi, self.log_phi = self.update_phi_fixed(doc_number, time, sslm, g3_matrix, g4_matrix, g5_matrix)
            lhood = self.compute_lda_lhood()
            converged = np.fabs((lhood_old - lhood) / (lhood_old * total))
        return lhood

    def update_lda_seq_ss(self, time, doc, topic_suffstats):
        """Update lda sequence sufficient statistics from an lda posterior.

        This is very similar to the :meth:`~gensim.models.ldaseqmodel.LdaPost.update_gamma` method and uses
        the same formula.

        Parameters
        ----------
        time : int
            The time slice.
        doc : list of (int, float)
            Unused but kept here for backwards compatibility. The document set in the constructor (`self.doc`) is used
            instead.
        topic_suffstats : list of float
            Sufficient statistics for each topic.

        Returns
        -------
        list of float
            The updated sufficient statistics for each topic.

        """
        num_topics = self.lda.num_topics
        for k in range(num_topics):
            topic_ss = topic_suffstats[k]
            n = 0
            for word_id, count in self.doc:
                topic_ss[word_id][time] += count * self.phi[n][k]
                n += 1
            topic_suffstats[k] = topic_ss
        return topic_suffstats