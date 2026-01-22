import re
import string
import warnings
from bleach._vendor.html5lib import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib import (
from bleach._vendor.html5lib.constants import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib.constants import (
from bleach._vendor.html5lib.filters.base import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib._inputstream import (
from bleach._vendor.html5lib.serializer import (
from bleach._vendor.html5lib._tokenizer import (
from bleach._vendor.html5lib._trie import (
class BleachHTMLTokenizer(HTMLTokenizer):
    """Tokenizer that doesn't consume character entities"""

    def __init__(self, consume_entities=False, **kwargs):
        super().__init__(**kwargs)
        self.consume_entities = consume_entities
        self.stream = InputStreamWithMemory(self.stream)
        self.emitted_last_token = None

    def __iter__(self):
        last_error_token = None
        for token in super().__iter__():
            if last_error_token is not None:
                if last_error_token['data'] == 'invalid-character-in-attribute-name' and token['type'] in TAG_TOKEN_TYPES and token.get('data'):
                    token['data'] = attributeMap(((attr_name, attr_value) for attr_name, attr_value in token['data'].items() if '"' not in attr_name and "'" not in attr_name and ('<' not in attr_name)))
                    last_error_token = None
                    yield token
                elif last_error_token['data'] == 'expected-closing-tag-but-got-char' and self.parser.tags is not None and (token['data'].lower().strip() not in self.parser.tags):
                    token['data'] = self.stream.get_tag()
                    token['type'] = TAG_TOKEN_TYPE_CHARACTERS
                    last_error_token = None
                    yield token
                elif token['type'] == TAG_TOKEN_TYPE_PARSEERROR:
                    yield last_error_token
                    last_error_token = token
                else:
                    yield last_error_token
                    yield token
                    last_error_token = None
                continue
            if token['type'] == TAG_TOKEN_TYPE_PARSEERROR:
                last_error_token = token
                continue
            yield token
        if last_error_token:
            if last_error_token['data'] == 'eof-in-tag-name':
                yield {'type': TAG_TOKEN_TYPE_CHARACTERS, 'data': self.stream.get_tag()}
            elif last_error_token['data'] in ('eof-in-attribute-name', 'eof-in-attribute-value-no-quotes'):
                yield {'type': TAG_TOKEN_TYPE_CHARACTERS, 'data': self.stream.get_tag()}
            else:
                yield last_error_token

    def consumeEntity(self, allowedChar=None, fromAttribute=False):
        if self.consume_entities:
            return super().consumeEntity(allowedChar, fromAttribute)
        if fromAttribute:
            self.currentToken['data'][-1][1] += '&'
        else:
            self.tokenQueue.append({'type': TAG_TOKEN_TYPE_CHARACTERS, 'data': '&'})

    def tagOpenState(self):
        self.stream.start_tag()
        return super().tagOpenState()

    def emitCurrentToken(self):
        token = self.currentToken
        if self.parser.tags is not None and token['type'] in TAG_TOKEN_TYPES and (token['name'].lower() not in self.parser.tags):
            if self.parser.strip:
                if self.emitted_last_token and token['type'] == TAG_TOKEN_TYPE_START and (token['name'].lower() in HTML_TAGS_BLOCK_LEVEL):
                    new_data = '\n'
                else:
                    new_data = ''
            else:
                new_data = self.stream.get_tag()
            new_token = {'type': TAG_TOKEN_TYPE_CHARACTERS, 'data': new_data}
            self.currentToken = self.emitted_last_token = new_token
            self.tokenQueue.append(new_token)
            self.state = self.dataState
            return
        self.emitted_last_token = self.currentToken
        super().emitCurrentToken()