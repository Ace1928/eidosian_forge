from markdownify import markdownify as md
def test_underscore():
    assert md('_hey_dude_') == '\\_hey\\_dude\\_'
    assert md('_hey_dude_', escape_underscores=False) == '_hey_dude_'