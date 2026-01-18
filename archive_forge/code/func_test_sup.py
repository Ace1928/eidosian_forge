from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_sup():
    assert md('<sup>foo</sup>') == 'foo'
    assert md('<sup>foo</sup>', sup_symbol='^') == '^foo^'