from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_strong_em_symbol():
    assert md('<strong>Hello</strong>', strong_em_symbol=UNDERSCORE) == '__Hello__'
    assert md('<b>Hello</b>', strong_em_symbol=UNDERSCORE) == '__Hello__'
    assert md('<em>Hello</em>', strong_em_symbol=UNDERSCORE) == '_Hello_'
    assert md('<i>Hello</i>', strong_em_symbol=UNDERSCORE) == '_Hello_'