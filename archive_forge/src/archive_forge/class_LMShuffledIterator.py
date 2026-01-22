import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
class LMShuffledIterator(object):

    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
        data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle else np.array(range(len(self.data)))
        for idx in epoch_indices:
            yield self.data[idx]

    @torch_only_method
    def stream_iterator(self, sent_stream):
        streams = [None] * self.bsz
        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)
        n_retain = 0
        while True:
            data[n_retain:].fill_(-1)
            target.fill_(-1)
            valid_batch = True
            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        data[n_retain + n_filled:n_retain + n_filled + n_new, i] = streams[i][:n_new]
                        target[n_filled:n_filled + n_new, i] = streams[i][1:n_new + 1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break
            if not valid_batch:
                return
            data_out = data.transpose(0, 1).contiguous().to(self.device)
            target_out = target.transpose(0, 1).contiguous().to(self.device)
            yield (data_out, target_out, self.bptt)
            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        sent_stream = self.get_sent_stream()
        for batch in self.stream_iterator(sent_stream):
            yield batch