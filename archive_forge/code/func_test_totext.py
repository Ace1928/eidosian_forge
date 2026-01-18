from __future__ import absolute_import, print_function, division
import io
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq, eq_
from petl.io.text import fromtext, totext
def test_totext():
    tbl = ((u'name', u'id'), (u'Արամ Խաչատրյան', 1), (u'Johann Strauß', 2), (u'Вагиф Сәмәдоғлу', 3), (u'章子怡', 4))
    prologue = u"{| class='wikitable'\n|-\n! name\n! id\n"
    template = u'|-\n| {name}\n| {id}\n'
    epilogue = u'|}\n'
    fn = NamedTemporaryFile().name
    totext(tbl, fn, template=template, prologue=prologue, epilogue=epilogue, encoding='utf-8')
    f = io.open(fn, encoding='utf-8', mode='rt')
    actual = f.read()
    expect = u"{| class='wikitable'\n|-\n! name\n! id\n|-\n| Արամ Խաչատրյան\n| 1\n|-\n| Johann Strauß\n| 2\n|-\n| Вагиф Сәмәдоғлу\n| 3\n|-\n| 章子怡\n| 4\n|}\n"
    eq_(expect, actual)