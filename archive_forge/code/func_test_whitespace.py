from markdownify import markdownify as md
def test_whitespace():
    assert md(' a  b \t\t c ') == ' a b c '