from io import BytesIO
from ... import tests
from .. import pack
def assertOutput(self, expected_output):
    """Assert that the output of self.writer ContainerWriter is equal to
        expected_output.
        """
    self.assertEqual(expected_output, self.output.getvalue())