import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def viterbi(self, sequence, state_alphabet):
    """Calculate the most probable state path using the Viterbi algorithm.

        This implements the Viterbi algorithm (see pgs 55-57 in Durbin et
        al for a full explanation -- this is where I took my implementation
        ideas from), to allow decoding of the state path, given a sequence
        of emissions.

        Arguments:
         - sequence -- A Seq object with the emission sequence that we
           want to decode.
         - state_alphabet -- An iterable (e.g., tuple or list) containing
           all of the letters that can appear in the states

        """
    log_initial = self._log_transform(self.initial_prob)
    log_trans = self._log_transform(self.transition_prob)
    log_emission = self._log_transform(self.emission_prob)
    viterbi_probs = {}
    pred_state_seq = {}
    for i in range(len(sequence)):
        for cur_state in state_alphabet:
            emission_part = log_emission[cur_state, sequence[i]]
            max_prob = 0
            if i == 0:
                max_prob = log_initial[cur_state]
            else:
                possible_state_probs = {}
                for prev_state in self.transitions_to(cur_state):
                    trans_part = log_trans[prev_state, cur_state]
                    viterbi_part = viterbi_probs[prev_state, i - 1]
                    cur_prob = viterbi_part + trans_part
                    possible_state_probs[prev_state] = cur_prob
                max_prob = max(possible_state_probs.values())
            viterbi_probs[cur_state, i] = emission_part + max_prob
            if i > 0:
                for state in possible_state_probs:
                    if possible_state_probs[state] == max_prob:
                        pred_state_seq[i - 1, cur_state] = state
                        break
    all_probs = {}
    for state in state_alphabet:
        all_probs[state] = viterbi_probs[state, len(sequence) - 1]
    state_path_prob = max(all_probs.values())
    last_state = ''
    for state in all_probs:
        if all_probs[state] == state_path_prob:
            last_state = state
    assert last_state != '', "Didn't find the last state to trace from!"
    traceback_seq = []
    loop_seq = list(range(1, len(sequence)))
    loop_seq.reverse()
    state = last_state
    traceback_seq.append(state)
    for i in loop_seq:
        state = pred_state_seq[i - 1, state]
        traceback_seq.append(state)
    traceback_seq.reverse()
    traceback_seq = ''.join(traceback_seq)
    return (Seq(traceback_seq), state_path_prob)