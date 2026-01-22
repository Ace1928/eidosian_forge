class TestDBHandler(object):
    """Test the DBHandler"""

    def setup_class(cls):
        """Setup this class"""
        cls.sequences = []

    def teardown_class(cls):
        """Remove our sequences"""
        for s in cls.sequences:
            try:
                s.delete()
            except:
                pass

    def test_sequence_generator_no_rollover(self):
        """Test the sequence generator without rollover"""
        from boto.sdb.db.sequence import SequenceGenerator
        gen = SequenceGenerator('ABC')
        assert gen('') == 'A'
        assert gen('A') == 'B'
        assert gen('B') == 'C'
        assert gen('C') == 'AA'
        assert gen('AC') == 'BA'

    def test_sequence_generator_with_rollover(self):
        """Test the sequence generator with rollover"""
        from boto.sdb.db.sequence import SequenceGenerator
        gen = SequenceGenerator('ABC', rollover=True)
        assert gen('') == 'A'
        assert gen('A') == 'B'
        assert gen('B') == 'C'
        assert gen('C') == 'A'

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

    def test_sequence_simple_string(self):
        from boto.sdb.db.sequence import Sequence, increment_string
        s = Sequence(fnc=increment_string)
        self.sequences.append(s)
        assert s.val == 'A'
        assert s.next() == 'B'

    def test_fib(self):
        """Test the fibonacci sequence generator"""
        from boto.sdb.db.sequence import fib
        lv = 0
        for v in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]:
            assert fib(v, lv) == lv + v
            lv = fib(v, lv)

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

    def test_sequence_string(self):
        """Test the String incrementation sequence"""
        from boto.sdb.db.sequence import Sequence, increment_string
        s = Sequence(fnc=increment_string)
        self.sequences.append(s)
        assert s.val == 'A'
        assert s.next() == 'B'
        s.val = 'Z'
        assert s.val == 'Z'
        assert s.next() == 'AA'