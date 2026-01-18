from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import json
from petl import fromjson, tojson
from petl.test.helpers import ieq
def test_tojson_2():
    table = [['name', 'wins'], ['Gilbert', [['straight', '7S'], ['one pair', '10H']]], ['Alexa', [['two pair', '4S'], ['two pair', '9S']]], ['May', []], ['Deloise', [['three of a kind', '5S']]]]
    f = NamedTemporaryFile(delete=False, mode='r')
    tojson(table, f.name, lines=True)
    result = []
    for line in f:
        result.append(json.loads(line))
    assert len(result) == 4
    assert result[0]['name'] == 'Gilbert'
    assert result[0]['wins'] == [['straight', '7S'], ['one pair', '10H']]
    assert result[1]['name'] == 'Alexa'
    assert result[1]['wins'] == [['two pair', '4S'], ['two pair', '9S']]
    assert result[2]['name'] == 'May'
    assert result[2]['wins'] == []
    assert result[3]['name'] == 'Deloise'
    assert result[3]['wins'] == [['three of a kind', '5S']]