from markdownify import markdownify as md
def test_ul():
    assert md('<ul><li>a</li><li>b</li></ul>') == '* a\n* b\n'
    assert md('<ul>\n     <li>\n             a\n     </li>\n     <li> b </li>\n     <li>   c\n     </li>\n </ul>') == '* a\n* b\n* c\n'