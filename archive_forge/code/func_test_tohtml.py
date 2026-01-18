from __future__ import absolute_import, print_function, division
import io
from tempfile import NamedTemporaryFile
from petl.test.helpers import eq_
from petl.io.html import tohtml
def test_tohtml():
    tbl = ((u'name', u'id'), (u'Արամ Խաչատրյան', 1), (u'Johann Strauß', 2), (u'Вагиф Сәмәдоғлу', 3), (u'章子怡', 4))
    fn = NamedTemporaryFile().name
    tohtml(tbl, fn, encoding='utf-8', lineterminator='\n')
    f = io.open(fn, mode='rt', encoding='utf-8', newline='')
    actual = f.read()
    expect = u"<table class='petl'>\n<thead>\n<tr>\n<th>name</th>\n<th>id</th>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td>Արամ Խաչատրյան</td>\n<td style='text-align: right'>1</td>\n</tr>\n<tr>\n<td>Johann Strauß</td>\n<td style='text-align: right'>2</td>\n</tr>\n<tr>\n<td>Вагиф Сәмәдоғлу</td>\n<td style='text-align: right'>3</td>\n</tr>\n<tr>\n<td>章子怡</td>\n<td style='text-align: right'>4</td>\n</tr>\n</tbody>\n</table>\n"
    eq_(expect, actual)