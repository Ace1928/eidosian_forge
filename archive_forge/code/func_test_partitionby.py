from toolz import first, identity, countby, partitionby
def test_partitionby():
    assert list(partitionby(identity, [])) == []
    vowels = 'aeiou'
    assert list(partitionby(vowels.__contains__, 'abcdefghi')) == [('a',), ('b', 'c', 'd'), ('e',), ('f', 'g', 'h'), ('i',)]
    assert list(map(first, partitionby(identity, [1, 1, 1, 2, 3, 3, 2, 2, 3]))) == [1, 2, 3, 2, 3]
    assert ''.join(map(first, partitionby(identity, 'Khhhaaaaannnnn!!!!'))) == 'Khan!'