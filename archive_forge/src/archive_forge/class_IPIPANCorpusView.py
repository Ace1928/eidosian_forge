import functools
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, concat
class IPIPANCorpusView(StreamBackedCorpusView):
    WORDS_MODE = 0
    SENTS_MODE = 1
    PARAS_MODE = 2

    def __init__(self, filename, startpos=0, **kwargs):
        StreamBackedCorpusView.__init__(self, filename, None, startpos, None)
        self.in_sentence = False
        self.position = 0
        self.show_tags = kwargs.pop('tags', True)
        self.disamb_only = kwargs.pop('disamb_only', True)
        self.mode = kwargs.pop('mode', IPIPANCorpusView.WORDS_MODE)
        self.simplify_tags = kwargs.pop('simplify_tags', False)
        self.one_tag = kwargs.pop('one_tag', True)
        self.append_no_space = kwargs.pop('append_no_space', False)
        self.append_space = kwargs.pop('append_space', False)
        self.replace_xmlentities = kwargs.pop('replace_xmlentities', True)

    def read_block(self, stream):
        sentence = []
        sentences = []
        space = False
        no_space = False
        tags = set()
        lines = self._read_data(stream)
        while True:
            if len(lines) <= 1:
                self._seek(stream)
                lines = self._read_data(stream)
            if lines == ['']:
                assert not sentences
                return []
            line = lines.pop()
            self.position += len(line) + 1
            if line.startswith('<chunk type="s"'):
                self.in_sentence = True
            elif line.startswith('<chunk type="p"'):
                pass
            elif line.startswith('<tok'):
                if self.append_space and space and (not no_space):
                    self._append_space(sentence)
                space = True
                no_space = False
                orth = ''
                tags = set()
            elif line.startswith('</chunk'):
                if self.in_sentence:
                    self.in_sentence = False
                    self._seek(stream)
                    if self.mode == self.SENTS_MODE:
                        return [sentence]
                    elif self.mode == self.WORDS_MODE:
                        if self.append_space:
                            self._append_space(sentence)
                        return sentence
                    else:
                        sentences.append(sentence)
                elif self.mode == self.PARAS_MODE:
                    self._seek(stream)
                    return [sentences]
            elif line.startswith('<orth'):
                orth = line[6:-7]
                if self.replace_xmlentities:
                    orth = orth.replace('&quot;', '"').replace('&amp;', '&')
            elif line.startswith('<lex'):
                if not self.disamb_only or line.find('disamb=') != -1:
                    tag = line[line.index('<ctag') + 6:line.index('</ctag')]
                    tags.add(tag)
            elif line.startswith('</tok'):
                if self.show_tags:
                    if self.simplify_tags:
                        tags = [t.split(':')[0] for t in tags]
                    if not self.one_tag or not self.disamb_only:
                        sentence.append((orth, tuple(tags)))
                    else:
                        sentence.append((orth, tags.pop()))
                else:
                    sentence.append(orth)
            elif line.startswith('<ns/>'):
                if self.append_space:
                    no_space = True
                if self.append_no_space:
                    if self.show_tags:
                        sentence.append(('', 'no-space'))
                    else:
                        sentence.append('')
            elif line.startswith('</cesAna'):
                pass

    def _read_data(self, stream):
        self.position = stream.tell()
        buff = stream.read(4096)
        lines = buff.split('\n')
        lines.reverse()
        return lines

    def _seek(self, stream):
        stream.seek(self.position)

    def _append_space(self, sentence):
        if self.show_tags:
            sentence.append((' ', 'space'))
        else:
            sentence.append(' ')