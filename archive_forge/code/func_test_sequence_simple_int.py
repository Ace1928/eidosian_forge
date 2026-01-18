def test_sequence_simple_int(self):
    """Test a simple counter sequence"""
    from boto.sdb.db.sequence import Sequence
    s = Sequence()
    self.sequences.append(s)
    assert s.val == 0
    assert s.next() == 1
    assert s.next() == 2
    s2 = Sequence(s.id)
    assert s2.val == 2
    assert s.next() == 3
    assert s.val == 3
    assert s2.val == 3