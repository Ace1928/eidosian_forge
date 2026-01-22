import json
import os
import tempfile
import warnings
from subprocess import PIPE
from nltk.internals import _java_options, config_java, find_jar, java
from nltk.parse.corenlp import CoreNLPParser
from nltk.tokenize.api import TokenizerI
class StanfordTokenizer(TokenizerI):
    """
    Interface to the Stanford Tokenizer

    >>> from nltk.tokenize.stanford import StanfordTokenizer
    >>> s = "Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\nThanks."
    >>> StanfordTokenizer().tokenize(s) # doctest: +SKIP
    ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.', 'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
    >>> s = "The colour of the wall is blue."
    >>> StanfordTokenizer(options={"americanize": True}).tokenize(s) # doctest: +SKIP
    ['The', 'color', 'of', 'the', 'wall', 'is', 'blue', '.']
    """
    _JAR = 'stanford-postagger.jar'

    def __init__(self, path_to_jar=None, encoding='utf8', options=None, verbose=False, java_options='-mx1000m'):
        warnings.warn(str("\nThe StanfordTokenizer will be deprecated in version 3.2.5.\nPlease use \x1b[91mnltk.parse.corenlp.CoreNLPParser\x1b[0m instead.'"), DeprecationWarning, stacklevel=2)
        self._stanford_jar = find_jar(self._JAR, path_to_jar, env_vars=('STANFORD_POSTAGGER',), searchpath=(), url=_stanford_url, verbose=verbose)
        self._encoding = encoding
        self.java_options = java_options
        options = {} if options is None else options
        self._options_cmd = ','.join((f'{key}={val}' for key, val in options.items()))

    @staticmethod
    def _parse_tokenized_output(s):
        return s.splitlines()

    def tokenize(self, s):
        """
        Use stanford tokenizer's PTBTokenizer to tokenize multiple sentences.
        """
        cmd = ['edu.stanford.nlp.process.PTBTokenizer']
        return self._parse_tokenized_output(self._execute(cmd, s))

    def _execute(self, cmd, input_, verbose=False):
        encoding = self._encoding
        cmd.extend(['-charset', encoding])
        _options_cmd = self._options_cmd
        if _options_cmd:
            cmd.extend(['-options', self._options_cmd])
        default_options = ' '.join(_java_options)
        config_java(options=self.java_options, verbose=verbose)
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as input_file:
            if isinstance(input_, str) and encoding:
                input_ = input_.encode(encoding)
            input_file.write(input_)
            input_file.flush()
            cmd.append(input_file.name)
            stdout, stderr = java(cmd, classpath=self._stanford_jar, stdout=PIPE, stderr=PIPE)
            stdout = stdout.decode(encoding)
        os.unlink(input_file.name)
        config_java(options=default_options, verbose=False)
        return stdout