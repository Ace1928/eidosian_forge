from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_strong():
    assert md('<strong>Hello</strong>') == '**Hello**'