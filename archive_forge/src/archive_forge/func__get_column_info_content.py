import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def _get_column_info_content(self, obs_freq, e_freq_table, log_base, random_expected):
    """Calculate the information content for a column (PRIVATE).

        Arguments:
         - obs_freq - The frequencies observed for each letter in the column.
         - e_freq_table - An optional argument specifying a dictionary with
           the expected frequencies for each letter.
         - log_base - The base of the logarithm to use in calculating the
           info content.

        """
    gap_char = '-'
    if e_freq_table:
        for key in obs_freq:
            if key != gap_char and key not in e_freq_table:
                raise ValueError(f'Frequency table provided does not contain observed letter {key}')
    total_info = 0.0
    for letter in obs_freq:
        inner_log = 0.0
        if letter != gap_char:
            if e_freq_table:
                inner_log = obs_freq[letter] / e_freq_table[letter]
            else:
                inner_log = obs_freq[letter] / random_expected
        if inner_log > 0:
            letter_info = obs_freq[letter] * math.log(inner_log) / math.log(log_base)
            total_info += letter_info
    return total_info