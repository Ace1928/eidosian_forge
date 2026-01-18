from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_pre():
    assert md('<pre>test\n    foo\nbar</pre>') == '\n```\ntest\n    foo\nbar\n```\n'
    assert md('<pre><code>test\n    foo\nbar</code></pre>') == '\n```\ntest\n    foo\nbar\n```\n'
    assert md('<pre>this_should_not_escape</pre>') == '\n```\nthis_should_not_escape\n```\n'