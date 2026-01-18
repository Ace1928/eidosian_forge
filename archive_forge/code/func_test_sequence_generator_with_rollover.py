def test_sequence_generator_with_rollover(self):
    """Test the sequence generator with rollover"""
    from boto.sdb.db.sequence import SequenceGenerator
    gen = SequenceGenerator('ABC', rollover=True)
    assert gen('') == 'A'
    assert gen('A') == 'B'
    assert gen('B') == 'C'
    assert gen('C') == 'A'