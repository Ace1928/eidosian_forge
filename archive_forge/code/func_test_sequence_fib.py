def test_sequence_fib(self):
    """Test the fibonacci sequence"""
    from boto.sdb.db.sequence import Sequence, fib
    s = Sequence(fnc=fib)
    s2 = Sequence(s.id)
    self.sequences.append(s)
    assert s.val == 1
    for v in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]:
        assert s.next() == v
        assert s.val == v
        assert s2.val == v