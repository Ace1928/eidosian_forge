from __future__ import absolute_import, print_function, division
import petl as etl
from petl.test.helpers import eq_
def test_repr_html_limit():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    etl.config.display_limit = 2
    expect = "<table class='petl'>\n<thead>\n<tr>\n<th>foo</th>\n<th>bar</th>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td>a</td>\n<td style='text-align: right'>1</td>\n</tr>\n<tr>\n<td>b</td>\n<td style='text-align: right'>2</td>\n</tr>\n</tbody>\n</table>\n<p><strong>...</strong></p>\n"
    actual = etl.wrap(table)._repr_html_()
    print(actual)
    for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
        eq_(l1, l2)