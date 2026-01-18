@given(port_numbers(allow_zero=True))
def test_port_numbers_bounds_allow_zero(self, port):
    """
            port_numbers(allow_zero=True) generates integers between 0 and
            65535, inclusive.
            """
    self.assertGreaterEqual(port, 0)
    self.assertLessEqual(port, 65535)