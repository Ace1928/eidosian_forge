from markdownify import markdownify as md
def test_single_tag():
    assert md('<span>Hello</span>') == 'Hello'