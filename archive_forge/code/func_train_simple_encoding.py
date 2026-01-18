import collections
from typing import Optional
import regex
import tiktoken
def train_simple_encoding():
    gpt2_pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    with open(__file__, 'r') as f:
        data = f.read()
    enc = SimpleBytePairEncoding.train(data, vocab_size=600, pat_str=gpt2_pattern)
    print("This is the sequence of merges performed in order to encode 'hello world':")
    tokens = enc.encode('hello world')
    assert enc.decode(tokens) == 'hello world'
    assert enc.decode_bytes(tokens) == b'hello world'
    assert enc.decode_tokens_bytes(tokens) == [b'hello', b' world']
    return enc