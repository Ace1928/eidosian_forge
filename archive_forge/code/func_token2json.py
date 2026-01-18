import re
import warnings
from contextlib import contextmanager
from ...processing_utils import ProcessorMixin
def token2json(self, tokens, is_inner_value=False, added_vocab=None):
    """
        Convert a (generated) token sequence into an ordered JSON format.
        """
    if added_vocab is None:
        added_vocab = self.tokenizer.get_added_vocab()
    output = {}
    while tokens:
        start_token = re.search('<s_(.*?)>', tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        key_escaped = re.escape(key)
        end_token = re.search(f'</s_{key_escaped}>', tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, '')
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(f'{start_token_escaped}(.*?){end_token_escaped}', tokens, re.IGNORECASE)
            if content is not None:
                content = content.group(1).strip()
                if '<s_' in content and '</s_' in content:
                    value = self.token2json(content, is_inner_value=True, added_vocab=added_vocab)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:
                    output[key] = []
                    for leaf in content.split('<sep/>'):
                        leaf = leaf.strip()
                        if leaf in added_vocab and leaf[0] == '<' and (leaf[-2:] == '/>'):
                            leaf = leaf[1:-2]
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]
            tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
            if tokens[:6] == '<sep/>':
                return [output] + self.token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)
    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {'text_sequence': tokens}