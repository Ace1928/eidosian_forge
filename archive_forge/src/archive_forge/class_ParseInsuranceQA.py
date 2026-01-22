from typing import Optional
import gzip
import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
class ParseInsuranceQA(object):
    version: Optional[str] = None
    label2answer_fname: Optional[str] = None

    @classmethod
    def read_gz(cls, filename):
        f = gzip.open(filename, 'rb')
        return [x.decode('utf-8') for x in f.readlines()]

    @classmethod
    def readlines(cls, path):
        if path.endswith('.gz'):
            lines = cls.read_gz(path)
        else:
            lines = open(path).readlines()
        return lines

    @classmethod
    def wids2sent(cls, wids, d_vocab):
        return ' '.join([d_vocab[w] for w in wids])

    @classmethod
    def read_vocab(cls, vocab_path):
        d_vocab = {}
        with open(vocab_path, 'r') as f:
            for line in f:
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 2:
                    raise ValueError('vocab file (%s) corrupted. Line (%s)' % (repr(line), vocab_path))
                else:
                    wid, word = fields
                    d_vocab[wid] = word
        return d_vocab

    @classmethod
    def read_label2answer(cls, label2answer_path_gz, d_vocab):
        lines = cls.readlines(label2answer_path_gz)
        d_label_answer = {}
        for line in lines:
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 2:
                raise ValueError('label2answer file (%s) corrupted. Line (%s)' % (repr(line), label2answer_path_gz))
            else:
                aid, s_wids = fields
                sent = cls.wids2sent(s_wids.split(), d_vocab)
                d_label_answer[aid] = sent
        return d_label_answer

    @classmethod
    def create_fb_format(cls, out_path, dtype, inpath, d_vocab, d_label_answer):
        pass

    @classmethod
    def write_data_files(cls, dpext, out_path, d_vocab, d_label_answer):
        pass

    @classmethod
    def build(cls, dpath):
        print('building version: %s' % cls.version)
        dpext = os.path.join(dpath, 'insuranceQA-master/%s' % cls.version)
        vocab_path = os.path.join(dpext, 'vocabulary')
        d_vocab = cls.read_vocab(vocab_path)
        label2answer_path_gz = os.path.join(dpext, cls.label2answer_fname)
        d_label_answer = cls.read_label2answer(label2answer_path_gz, d_vocab)
        out_path = os.path.join(dpath, cls.version)
        build_data.make_dir(out_path)
        cls.write_data_files(dpext, out_path, d_vocab, d_label_answer)