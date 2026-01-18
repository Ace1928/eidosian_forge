from io import BytesIO
from twisted.python.htmlizer import filter
from twisted.trial.unittest import TestCase
def test_variable(self) -> None:
    """
        If passed an input file containing a variable access, L{filter} writes
        a I{pre} tag containing a I{py-src-variable} span containing the
        variable.
        """
    input = BytesIO(b'foo\n')
    output = BytesIO()
    filter(input, output)
    self.assertEqual(output.getvalue(), b'<pre><span class="py-src-variable">foo</span><span class="py-src-newline">\n</span><span class="py-src-endmarker"></span></pre>\n')