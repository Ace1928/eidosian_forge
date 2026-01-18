import tokenize
Return source code based on tokens.

    This is like tokenize.untokenize(), but it preserves spacing between
    tokens. So if the original soure code had multiple spaces between some
    tokens or if escaped newlines were used, those things will be reflected
    by untokenize().

    