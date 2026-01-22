from math import log
from operator import itemgetter
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI
def _tagword(self, sent, current_states):
    """
        :param sent : List of words remaining in the sentence
        :type sent  : [word,]
        :param current_states : List of possible tag combinations for
                                the sentence so far, and the log probability
                                associated with each tag combination
        :type current_states  : [([tag, ], logprob), ]

        Tags the first word in the sentence and
        recursively tags the reminder of sentence

        Uses formula specified above to calculate the probability
        of a particular tag
        """
    if sent == []:
        h, logp = current_states[0]
        return h
    word = sent[0]
    sent = sent[1:]
    new_states = []
    C = False
    if self._C and word[0].isupper():
        C = True
    if word in self._wd:
        self.known += 1
        for history, curr_sent_logprob in current_states:
            logprobs = []
            for t in self._wd[word].keys():
                tC = (t, C)
                p_uni = self._uni.freq(tC)
                p_bi = self._bi[history[-1]].freq(tC)
                p_tri = self._tri[tuple(history[-2:])].freq(tC)
                p_wd = self._wd[word][t] / self._uni[tC]
                p = self._l1 * p_uni + self._l2 * p_bi + self._l3 * p_tri
                p2 = log(p, 2) + log(p_wd, 2)
                new_states.append((history + [tC], curr_sent_logprob + p2))
    else:
        self.unknown += 1
        p = 1
        if self._unk is None:
            tag = ('Unk', C)
        else:
            [(_w, t)] = list(self._unk.tag([word]))
            tag = (t, C)
        for history, logprob in current_states:
            history.append(tag)
        new_states = current_states
    new_states.sort(reverse=True, key=itemgetter(1))
    if len(new_states) > self._N:
        new_states = new_states[:self._N]
    return self._tagword(sent, new_states)