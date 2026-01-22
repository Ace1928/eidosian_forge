from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
class HiddenMarkovModelTagger(TaggerI):
    """
    Hidden Markov model class, a generative model for labelling sequence data.
    These models define the joint probability of a sequence of symbols and
    their labels (state transitions) as the product of the starting state
    probability, the probability of each state transition, and the probability
    of each observation being generated from each state. This is described in
    more detail in the module documentation.

    This implementation is based on the HMM description in Chapter 8, Huang,
    Acero and Hon, Spoken Language Processing and includes an extension for
    training shallow HMM parsers or specialized HMMs as in Molina et.
    al, 2002.  A specialized HMM modifies training data by applying a
    specialization function to create a new training set that is more
    appropriate for sequential tagging with an HMM.  A typical use case is
    chunking.

    :param symbols: the set of output symbols (alphabet)
    :type symbols: seq of any
    :param states: a set of states representing state space
    :type states: seq of any
    :param transitions: transition probabilities; Pr(s_i | s_j) is the
        probability of transition from state i given the model is in
        state_j
    :type transitions: ConditionalProbDistI
    :param outputs: output probabilities; Pr(o_k | s_i) is the probability
        of emitting symbol k when entering state i
    :type outputs: ConditionalProbDistI
    :param priors: initial state distribution; Pr(s_i) is the probability
        of starting in state i
    :type priors: ProbDistI
    :param transform: an optional function for transforming training
        instances, defaults to the identity function.
    :type transform: callable
    """

    def __init__(self, symbols, states, transitions, outputs, priors, transform=_identity):
        self._symbols = unique_list(symbols)
        self._states = unique_list(states)
        self._transitions = transitions
        self._outputs = outputs
        self._priors = priors
        self._cache = None
        self._transform = transform

    @classmethod
    def _train(cls, labeled_sequence, test_sequence=None, unlabeled_sequence=None, transform=_identity, estimator=None, **kwargs):
        if estimator is None:

            def estimator(fd, bins):
                return LidstoneProbDist(fd, 0.1, bins)
        labeled_sequence = LazyMap(transform, labeled_sequence)
        symbols = unique_list((word for sent in labeled_sequence for word, tag in sent))
        tag_set = unique_list((tag for sent in labeled_sequence for word, tag in sent))
        trainer = HiddenMarkovModelTrainer(tag_set, symbols)
        hmm = trainer.train_supervised(labeled_sequence, estimator=estimator)
        hmm = cls(hmm._symbols, hmm._states, hmm._transitions, hmm._outputs, hmm._priors, transform=transform)
        if test_sequence:
            hmm.test(test_sequence, verbose=kwargs.get('verbose', False))
        if unlabeled_sequence:
            max_iterations = kwargs.get('max_iterations', 5)
            hmm = trainer.train_unsupervised(unlabeled_sequence, model=hmm, max_iterations=max_iterations)
            if test_sequence:
                hmm.test(test_sequence, verbose=kwargs.get('verbose', False))
        return hmm

    @classmethod
    def train(cls, labeled_sequence, test_sequence=None, unlabeled_sequence=None, **kwargs):
        """
        Train a new HiddenMarkovModelTagger using the given labeled and
        unlabeled training instances. Testing will be performed if test
        instances are provided.

        :return: a hidden markov model tagger
        :rtype: HiddenMarkovModelTagger
        :param labeled_sequence: a sequence of labeled training instances,
            i.e. a list of sentences represented as tuples
        :type labeled_sequence: list(list)
        :param test_sequence: a sequence of labeled test instances
        :type test_sequence: list(list)
        :param unlabeled_sequence: a sequence of unlabeled training instances,
            i.e. a list of sentences represented as words
        :type unlabeled_sequence: list(list)
        :param transform: an optional function for transforming training
            instances, defaults to the identity function, see ``transform()``
        :type transform: function
        :param estimator: an optional function or class that maps a
            condition's frequency distribution to its probability
            distribution, defaults to a Lidstone distribution with gamma = 0.1
        :type estimator: class or function
        :param verbose: boolean flag indicating whether training should be
            verbose or include printed output
        :type verbose: bool
        :param max_iterations: number of Baum-Welch iterations to perform
        :type max_iterations: int
        """
        return cls._train(labeled_sequence, test_sequence, unlabeled_sequence, **kwargs)

    def probability(self, sequence):
        """
        Returns the probability of the given symbol sequence. If the sequence
        is labelled, then returns the joint probability of the symbol, state
        sequence. Otherwise, uses the forward algorithm to find the
        probability over all label sequences.

        :return: the probability of the sequence
        :rtype: float
        :param sequence: the sequence of symbols which must contain the TEXT
            property, and optionally the TAG property
        :type sequence:  Token
        """
        return 2 ** self.log_probability(self._transform(sequence))

    def log_probability(self, sequence):
        """
        Returns the log-probability of the given symbol sequence. If the
        sequence is labelled, then returns the joint log-probability of the
        symbol, state sequence. Otherwise, uses the forward algorithm to find
        the log-probability over all label sequences.

        :return: the log-probability of the sequence
        :rtype: float
        :param sequence: the sequence of symbols which must contain the TEXT
            property, and optionally the TAG property
        :type sequence:  Token
        """
        sequence = self._transform(sequence)
        T = len(sequence)
        if T > 0 and sequence[0][_TAG]:
            last_state = sequence[0][_TAG]
            p = self._priors.logprob(last_state) + self._output_logprob(last_state, sequence[0][_TEXT])
            for t in range(1, T):
                state = sequence[t][_TAG]
                p += self._transitions[last_state].logprob(state) + self._output_logprob(state, sequence[t][_TEXT])
                last_state = state
            return p
        else:
            alpha = self._forward_probability(sequence)
            p = logsumexp2(alpha[T - 1])
            return p

    def tag(self, unlabeled_sequence):
        """
        Tags the sequence with the highest probability state sequence. This
        uses the best_path method to find the Viterbi path.

        :return: a labelled sequence of symbols
        :rtype: list
        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)
        return self._tag(unlabeled_sequence)

    def _tag(self, unlabeled_sequence):
        path = self._best_path(unlabeled_sequence)
        return list(zip(unlabeled_sequence, path))

    def _output_logprob(self, state, symbol):
        """
        :return: the log probability of the symbol being observed in the given
            state
        :rtype: float
        """
        return self._outputs[state].logprob(symbol)

    def _create_cache(self):
        """
        The cache is a tuple (P, O, X, S) where:

          - S maps symbols to integers.  I.e., it is the inverse
            mapping from self._symbols; for each symbol s in
            self._symbols, the following is true::

              self._symbols[S[s]] == s

          - O is the log output probabilities::

              O[i,k] = log( P(token[t]=sym[k]|tag[t]=state[i]) )

          - X is the log transition probabilities::

              X[i,j] = log( P(tag[t]=state[j]|tag[t-1]=state[i]) )

          - P is the log prior probabilities::

              P[i] = log( P(tag[0]=state[i]) )
        """
        if not self._cache:
            N = len(self._states)
            M = len(self._symbols)
            P = np.zeros(N, np.float32)
            X = np.zeros((N, N), np.float32)
            O = np.zeros((N, M), np.float32)
            for i in range(N):
                si = self._states[i]
                P[i] = self._priors.logprob(si)
                for j in range(N):
                    X[i, j] = self._transitions[si].logprob(self._states[j])
                for k in range(M):
                    O[i, k] = self._output_logprob(si, self._symbols[k])
            S = {}
            for k in range(M):
                S[self._symbols[k]] = k
            self._cache = (P, O, X, S)

    def _update_cache(self, symbols):
        if symbols:
            self._create_cache()
            P, O, X, S = self._cache
            for symbol in symbols:
                if symbol not in self._symbols:
                    self._cache = None
                    self._symbols.append(symbol)
            if not self._cache:
                N = len(self._states)
                M = len(self._symbols)
                Q = O.shape[1]
                O = np.hstack([O, np.zeros((N, M - Q), np.float32)])
                for i in range(N):
                    si = self._states[i]
                    for k in range(Q, M):
                        O[i, k] = self._output_logprob(si, self._symbols[k])
                for k in range(Q, M):
                    S[self._symbols[k]] = k
                self._cache = (P, O, X, S)

    def reset_cache(self):
        self._cache = None

    def best_path(self, unlabeled_sequence):
        """
        Returns the state sequence of the optimal (most probable) path through
        the HMM. Uses the Viterbi algorithm to calculate this part by dynamic
        programming.

        :return: the state sequence
        :rtype: sequence of any
        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)
        return self._best_path(unlabeled_sequence)

    def _best_path(self, unlabeled_sequence):
        T = len(unlabeled_sequence)
        N = len(self._states)
        self._create_cache()
        self._update_cache(unlabeled_sequence)
        P, O, X, S = self._cache
        V = np.zeros((T, N), np.float32)
        B = -np.ones((T, N), int)
        V[0] = P + O[:, S[unlabeled_sequence[0]]]
        for t in range(1, T):
            for j in range(N):
                vs = V[t - 1, :] + X[:, j]
                best = np.argmax(vs)
                V[t, j] = vs[best] + O[j, S[unlabeled_sequence[t]]]
                B[t, j] = best
        current = np.argmax(V[T - 1, :])
        sequence = [current]
        for t in range(T - 1, 0, -1):
            last = B[t, current]
            sequence.append(last)
            current = last
        sequence.reverse()
        return list(map(self._states.__getitem__, sequence))

    def best_path_simple(self, unlabeled_sequence):
        """
        Returns the state sequence of the optimal (most probable) path through
        the HMM. Uses the Viterbi algorithm to calculate this part by dynamic
        programming.  This uses a simple, direct method, and is included for
        teaching purposes.

        :return: the state sequence
        :rtype: sequence of any
        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)
        return self._best_path_simple(unlabeled_sequence)

    def _best_path_simple(self, unlabeled_sequence):
        T = len(unlabeled_sequence)
        N = len(self._states)
        V = np.zeros((T, N), np.float64)
        B = {}
        symbol = unlabeled_sequence[0]
        for i, state in enumerate(self._states):
            V[0, i] = self._priors.logprob(state) + self._output_logprob(state, symbol)
            B[0, state] = None
        for t in range(1, T):
            symbol = unlabeled_sequence[t]
            for j in range(N):
                sj = self._states[j]
                best = None
                for i in range(N):
                    si = self._states[i]
                    va = V[t - 1, i] + self._transitions[si].logprob(sj)
                    if not best or va > best[0]:
                        best = (va, si)
                V[t, j] = best[0] + self._output_logprob(sj, symbol)
                B[t, sj] = best[1]
        best = None
        for i in range(N):
            val = V[T - 1, i]
            if not best or val > best[0]:
                best = (val, self._states[i])
        current = best[1]
        sequence = [current]
        for t in range(T - 1, 0, -1):
            last = B[t, current]
            sequence.append(last)
            current = last
        sequence.reverse()
        return sequence

    def random_sample(self, rng, length):
        """
        Randomly sample the HMM to generate a sentence of a given length. This
        samples the prior distribution then the observation distribution and
        transition distribution for each subsequent observation and state.
        This will mostly generate unintelligible garbage, but can provide some
        amusement.

        :return:        the randomly created state/observation sequence,
                        generated according to the HMM's probability
                        distributions. The SUBTOKENS have TEXT and TAG
                        properties containing the observation and state
                        respectively.
        :rtype:         list
        :param rng:     random number generator
        :type rng:      Random (or any object with a random() method)
        :param length:  desired output length
        :type length:   int
        """
        tokens = []
        state = self._sample_probdist(self._priors, rng.random(), self._states)
        symbol = self._sample_probdist(self._outputs[state], rng.random(), self._symbols)
        tokens.append((symbol, state))
        for i in range(1, length):
            state = self._sample_probdist(self._transitions[state], rng.random(), self._states)
            symbol = self._sample_probdist(self._outputs[state], rng.random(), self._symbols)
            tokens.append((symbol, state))
        return tokens

    def _sample_probdist(self, probdist, p, samples):
        cum_p = 0
        for sample in samples:
            add_p = probdist.prob(sample)
            if cum_p <= p <= cum_p + add_p:
                return sample
            cum_p += add_p
        raise Exception('Invalid probability distribution - does not sum to one')

    def entropy(self, unlabeled_sequence):
        """
        Returns the entropy over labellings of the given sequence. This is
        given by::

            H(O) = - sum_S Pr(S | O) log Pr(S | O)

        where the summation ranges over all state sequences, S. Let
        *Z = Pr(O) = sum_S Pr(S, O)}* where the summation ranges over all state
        sequences and O is the observation sequence. As such the entropy can
        be re-expressed as::

            H = - sum_S Pr(S | O) log [ Pr(S, O) / Z ]
            = log Z - sum_S Pr(S | O) log Pr(S, 0)
            = log Z - sum_S Pr(S | O) [ log Pr(S_0) + sum_t Pr(S_t | S_{t-1}) + sum_t Pr(O_t | S_t) ]

        The order of summation for the log terms can be flipped, allowing
        dynamic programming to be used to calculate the entropy. Specifically,
        we use the forward and backward probabilities (alpha, beta) giving::

            H = log Z - sum_s0 alpha_0(s0) beta_0(s0) / Z * log Pr(s0)
            + sum_t,si,sj alpha_t(si) Pr(sj | si) Pr(O_t+1 | sj) beta_t(sj) / Z * log Pr(sj | si)
            + sum_t,st alpha_t(st) beta_t(st) / Z * log Pr(O_t | st)

        This simply uses alpha and beta to find the probabilities of partial
        sequences, constrained to include the given state(s) at some point in
        time.
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)
        T = len(unlabeled_sequence)
        N = len(self._states)
        alpha = self._forward_probability(unlabeled_sequence)
        beta = self._backward_probability(unlabeled_sequence)
        normalisation = logsumexp2(alpha[T - 1])
        entropy = normalisation
        for i, state in enumerate(self._states):
            p = 2 ** (alpha[0, i] + beta[0, i] - normalisation)
            entropy -= p * self._priors.logprob(state)
        for t0 in range(T - 1):
            t1 = t0 + 1
            for i0, s0 in enumerate(self._states):
                for i1, s1 in enumerate(self._states):
                    p = 2 ** (alpha[t0, i0] + self._transitions[s0].logprob(s1) + self._outputs[s1].logprob(unlabeled_sequence[t1][_TEXT]) + beta[t1, i1] - normalisation)
                    entropy -= p * self._transitions[s0].logprob(s1)
        for t in range(T):
            for i, state in enumerate(self._states):
                p = 2 ** (alpha[t, i] + beta[t, i] - normalisation)
                entropy -= p * self._outputs[state].logprob(unlabeled_sequence[t][_TEXT])
        return entropy

    def point_entropy(self, unlabeled_sequence):
        """
        Returns the pointwise entropy over the possible states at each
        position in the chain, given the observation sequence.
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)
        T = len(unlabeled_sequence)
        N = len(self._states)
        alpha = self._forward_probability(unlabeled_sequence)
        beta = self._backward_probability(unlabeled_sequence)
        normalisation = logsumexp2(alpha[T - 1])
        entropies = np.zeros(T, np.float64)
        probs = np.zeros(N, np.float64)
        for t in range(T):
            for s in range(N):
                probs[s] = alpha[t, s] + beta[t, s] - normalisation
            for s in range(N):
                entropies[t] -= 2 ** probs[s] * probs[s]
        return entropies

    def _exhaustive_entropy(self, unlabeled_sequence):
        unlabeled_sequence = self._transform(unlabeled_sequence)
        T = len(unlabeled_sequence)
        N = len(self._states)
        labellings = [[state] for state in self._states]
        for t in range(T - 1):
            current = labellings
            labellings = []
            for labelling in current:
                for state in self._states:
                    labellings.append(labelling + [state])
        log_probs = []
        for labelling in labellings:
            labeled_sequence = unlabeled_sequence[:]
            for t, label in enumerate(labelling):
                labeled_sequence[t] = (labeled_sequence[t][_TEXT], label)
            lp = self.log_probability(labeled_sequence)
            log_probs.append(lp)
        normalisation = _log_add(*log_probs)
        entropy = 0
        for lp in log_probs:
            lp -= normalisation
            entropy -= 2 ** lp * lp
        return entropy

    def _exhaustive_point_entropy(self, unlabeled_sequence):
        unlabeled_sequence = self._transform(unlabeled_sequence)
        T = len(unlabeled_sequence)
        N = len(self._states)
        labellings = [[state] for state in self._states]
        for t in range(T - 1):
            current = labellings
            labellings = []
            for labelling in current:
                for state in self._states:
                    labellings.append(labelling + [state])
        log_probs = []
        for labelling in labellings:
            labelled_sequence = unlabeled_sequence[:]
            for t, label in enumerate(labelling):
                labelled_sequence[t] = (labelled_sequence[t][_TEXT], label)
            lp = self.log_probability(labelled_sequence)
            log_probs.append(lp)
        normalisation = _log_add(*log_probs)
        probabilities = _ninf_array((T, N))
        for labelling, lp in zip(labellings, log_probs):
            lp -= normalisation
            for t, label in enumerate(labelling):
                index = self._states.index(label)
                probabilities[t, index] = _log_add(probabilities[t, index], lp)
        entropies = np.zeros(T, np.float64)
        for t in range(T):
            for s in range(N):
                entropies[t] -= 2 ** probabilities[t, s] * probabilities[t, s]
        return entropies

    def _transitions_matrix(self):
        """Return a matrix of transition log probabilities."""
        trans_iter = (self._transitions[sj].logprob(si) for sj in self._states for si in self._states)
        transitions_logprob = np.fromiter(trans_iter, dtype=np.float64)
        N = len(self._states)
        return transitions_logprob.reshape((N, N)).T

    def _outputs_vector(self, symbol):
        """
        Return a vector with log probabilities of emitting a symbol
        when entering states.
        """
        out_iter = (self._output_logprob(sj, symbol) for sj in self._states)
        return np.fromiter(out_iter, dtype=np.float64)

    def _forward_probability(self, unlabeled_sequence):
        """
        Return the forward probability matrix, a T by N array of
        log-probabilities, where T is the length of the sequence and N is the
        number of states. Each entry (t, s) gives the probability of being in
        state s at time t after observing the partial symbol sequence up to
        and including t.

        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        :return: the forward log probability matrix
        :rtype: array
        """
        T = len(unlabeled_sequence)
        N = len(self._states)
        alpha = _ninf_array((T, N))
        transitions_logprob = self._transitions_matrix()
        symbol = unlabeled_sequence[0][_TEXT]
        for i, state in enumerate(self._states):
            alpha[0, i] = self._priors.logprob(state) + self._output_logprob(state, symbol)
        for t in range(1, T):
            symbol = unlabeled_sequence[t][_TEXT]
            output_logprob = self._outputs_vector(symbol)
            for i in range(N):
                summand = alpha[t - 1] + transitions_logprob[i]
                alpha[t, i] = logsumexp2(summand) + output_logprob[i]
        return alpha

    def _backward_probability(self, unlabeled_sequence):
        """
        Return the backward probability matrix, a T by N array of
        log-probabilities, where T is the length of the sequence and N is the
        number of states. Each entry (t, s) gives the probability of being in
        state s at time t after observing the partial symbol sequence from t
        .. T.

        :return: the backward log probability matrix
        :rtype:  array
        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        """
        T = len(unlabeled_sequence)
        N = len(self._states)
        beta = _ninf_array((T, N))
        transitions_logprob = self._transitions_matrix().T
        beta[T - 1, :] = np.log2(1)
        for t in range(T - 2, -1, -1):
            symbol = unlabeled_sequence[t + 1][_TEXT]
            outputs = self._outputs_vector(symbol)
            for i in range(N):
                summand = transitions_logprob[i] + beta[t + 1] + outputs
                beta[t, i] = logsumexp2(summand)
        return beta

    def test(self, test_sequence, verbose=False, **kwargs):
        """
        Tests the HiddenMarkovModelTagger instance.

        :param test_sequence: a sequence of labeled test instances
        :type test_sequence: list(list)
        :param verbose: boolean flag indicating whether training should be
            verbose or include printed output
        :type verbose: bool
        """

        def words(sent):
            return [word for word, tag in sent]

        def tags(sent):
            return [tag for word, tag in sent]

        def flatten(seq):
            return list(itertools.chain(*seq))
        test_sequence = self._transform(test_sequence)
        predicted_sequence = list(map(self._tag, map(words, test_sequence)))
        if verbose:
            for test_sent, predicted_sent in zip(test_sequence, predicted_sequence):
                print('Test:', ' '.join((f'{token}/{tag}' for token, tag in test_sent)))
                print()
                print('Untagged:', ' '.join(('%s' % token for token, tag in test_sent)))
                print()
                print('HMM-tagged:', ' '.join((f'{token}/{tag}' for token, tag in predicted_sent)))
                print()
                print('Entropy:', self.entropy([(token, None) for token, tag in predicted_sent]))
                print()
                print('-' * 60)
        test_tags = flatten(map(tags, test_sequence))
        predicted_tags = flatten(map(tags, predicted_sequence))
        acc = accuracy(test_tags, predicted_tags)
        count = sum((len(sent) for sent in test_sequence))
        print('accuracy over %d tokens: %.2f' % (count, acc * 100))

    def __repr__(self):
        return '<HiddenMarkovModelTagger %d states and %d output symbols>' % (len(self._states), len(self._symbols))