import collections
import json
import os
import re
from typing import Optional, Tuple
import numpy as np
from ...tokenization_utils_fast import PreTrainedTokenizer
from ...utils import logging
class SubWordJapaneseTokenizer(object):
    """
    https://github.com/tanreinama/Japanese-BPEEncoder_V2 This tokenizer class is under MIT Lisence according to the
    original repository.

    MIT License

    Copyright (c) 2020 tanreinama

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, vocab, ids_to_tokens, emoji):
        self.vocab = vocab
        self.ids_to_tokens = ids_to_tokens
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.vocab.keys()])
        self.content_repatter1 = re.compile("(https?|ftp)(:\\/\\/[-_\\.!~*\\'()a-zA-Z0-9;\\/?:\\@&=\\+$,%#]+)")
        self.content_repatter2 = re.compile('[A-Za-z0-9\\._+]*@[\\-_0-9A-Za-z]+(\\.[A-Za-z]+)*')
        self.content_repatter3 = re.compile('[\\(]{0,1}[0-9]{2,4}[\\)\\-\\(]{0,1}[0-9]{2,4}[\\)\\-]{0,1}[0-9]{3,4}')
        self.content_repatter4 = re.compile('([12]\\d{3}[/\\-年])*(0?[1-9]|1[0-2])[/\\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\\d{1,2}|:|\\d{1,2}時|\\d{1,2}分|\\(日\\)|\\(月\\)|\\(火\\)|\\(水\\)|\\(木\\)|\\(金\\)|\\(土\\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*')
        self.content_repatter5 = re.compile('(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\\u32ff)\\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\\d{1,2}|:|\\d{1,2}時|\\d{1,2}分|\\(日\\)|\\(月\\)|\\(火\\)|\\(水\\)|\\(木\\)|\\(金\\)|\\(土\\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*')
        self.content_repatter6 = re.compile('((0|[1-9]\\d*|[1-9]\\d{0,2}(,\\d{3})+)*億)*((0|[1-9]\\d*|[1-9]\\d{0,2}(,\\d{3})+)*万)*((0|[1-9]\\d*|[1-9]\\d{0,2}(,\\d{3})+)*千)*(0|[1-9]\\d*|[1-9]\\d{0,2}(,\\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\\(税込\\)|\\(税抜\\)|\\+tax)*')
        keisen = '─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿'
        blocks = '▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟'
        self.content_trans1 = str.maketrans({k: '<BLOCK>' for k in keisen + blocks})

    def __len__(self):
        return len(self.ids_to_tokens)

    def clean_text(self, content):
        content = self.content_repatter1.sub('<URL>', content)
        content = self.content_repatter2.sub('<EMAIL>', content)
        content = self.content_repatter3.sub('<TEL>', content)
        content = self.content_repatter4.sub('<DATE>', content)
        content = self.content_repatter5.sub('<DATE>', content)
        content = self.content_repatter6.sub('<PRICE>', content)
        content = content.translate(self.content_trans1)
        while '<BLOCK><BLOCK>' in content:
            content = content.replace('<BLOCK><BLOCK>', '<BLOCK>')
        return content

    def tokenize(self, text, clean=False):
        text = text.replace(' ', '<SP>')
        text = text.replace('\u3000', '<SP>')
        text = text.replace('\r\n', '<BR>')
        text = text.replace('\n', '<BR>')
        text = text.replace('\r', '<BR>')
        text = text.replace('\t', '<TAB>')
        text = text.replace('—', 'ー')
        text = text.replace('−', 'ー')
        for k, v in self.emoji['emoji'].items():
            if k in text:
                text = text.replace(k, v)
        if clean:
            text = self.clean_text(text)

        def check_simbol(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 2:
                c = (int(e[0]) << 8) + int(e[1])
                if c >= 49825 and c <= 49855 or (c >= 51072 and c <= 51075) or (c >= 51897 and c <= 52159) or (c >= 52352 and c <= 52642):
                    return True
            return False

        def checku2e(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 3:
                c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
                if c >= 14844032 and c <= 14856319:
                    return True
            return False
        pos = 0
        result = []
        while pos < len(text):
            end = min(len(text), pos + self.maxlen + 1) if text[pos] == '<' else pos + 3
            candidates = []
            for e in range(end, pos, -1):
                wd = text[pos:e]
                if wd in self.vocab:
                    if wd[0] == '<' and len(wd) > 2:
                        candidates = [(self.vocab[wd], wd, e)]
                        break
                    else:
                        candidates.append((self.vocab[wd], wd, e))
            if len(candidates) > 0:
                _, wd, e = sorted(candidates, key=lambda x: x[0])[0]
                result.append(wd)
                pos = e
            else:
                end = pos + 1
                wd = text[pos:end]
                if check_simbol(wd):
                    result.append('<KIGOU>')
                elif checku2e(wd):
                    result.append('<U2000U2BFF>')
                else:
                    for i in wd.encode('utf-8'):
                        result.append('<|byte%d|>' % i)
                pos = end
        return result

    def convert_id_to_token(self, index, breakline='\n'):
        words = []
        byte_tokens = []
        word = self.ids_to_tokens[index][0]
        if word[:6] == '<|byte' and word[-2:] == '|>':
            byte_tokens.append(int(word[6:-2]))
        else:
            if len(byte_tokens) > 0:
                words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
                byte_tokens = []
            if word[:7] == '<|emoji' and word[-2:] == '|>':
                words.append(self.emoji['emoji_inv'][word])
            elif word == '<SP>':
                words.append(' ')
            elif word == '<BR>':
                words.append(breakline)
            elif word == '<TAB>':
                words.append('\t')
            elif word == '<BLOCK>':
                words.append('▀')
            elif word == '<KIGOU>':
                words.append('ǀ')
            elif word == '<U2000U2BFF>':
                words.append('‖')
            else:
                words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
        text = ''.join(words)
        return text