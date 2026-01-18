from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_rowreduce_fieldnameaccess():
    table1 = (('foo', 'bar'), ('a', 3), ('a', 7), ('b', 2), ('b', 1), ('b', 9), ('c', 4))

    def sumbar(key, records):
        return [key, sum([rec['bar'] for rec in records])]
    table2 = rowreduce(table1, key='foo', reducer=sumbar, header=['foo', 'barsum'])
    expect2 = (('foo', 'barsum'), ('a', 10), ('b', 12), ('c', 4))
    ieq(expect2, table2)